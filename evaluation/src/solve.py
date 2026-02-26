import hydra
import json
from omegaconf import OmegaConf
import ray
from torch.utils.data import DataLoader, SequentialSampler
from pathlib import Path

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls

from azr_minor.rewards.custom_evaluate import extract_code
from azr_minor.utils.code_utils.sandboxfusion_executor import SandboxfusionExecutor

Language = {
    "python": "Python",
    "nodejs": "Javascript",
    "cpp": "C++",
    "java": "Java",
    "go": "Go",
    "julia": "Julia",
    "rust": "Rust",
    "racket": "Racket",
}
comment_out = {
    "python": "#",
    "nodejs": "//",
    "cpp": "//",
    "java": "//",
    "go": "//",
    "julia": "#",
    "rust": "//",
    "racket": ";"
}

instruction_prefix = {
    "instruct": "Please provide a self-contained {Language} script that solves the following problem in a markdown code block:",
    "perf-instruct": "Please provide an efficient and self-contained {Language} script that solves the following problem in a markdown code block:",
    "perf-CoT": "Think step by step: please provide an efficient and self-contained {Language} script that solves the following problem in a markdown code block:",
    "azr": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Please provide an efficient and self-contained {Language} script that solves the following problem in a markdown code block:"
}

response_prefix = {
    "instruct": "Below is a {Language} script with a self-contained function that solves the problem and passes corresponding tests:",
    "perf-instruct": "Below is a {Language} script with a self-contained function that efficiently solves the problem and passes corresponding tests:",
    "perf-CoT": "Below is a {Language} script with a self-contained function that efficiently solves the problem and passes corresponding tests:",
    "azr": "Assistant: <think>"
}

instruction = """# Task: You will be given a question (problem specification) and will generate a correct {Language} program that matches the specification and passes all tests. Your final answer should be wrapped in ```{language}``` tags.
Question: {question}
    
You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```{language}
{comment_out} YOUR CODE HERE
```
        
Assistant: <think>
"""
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


task_prompt = """\
{instruction_prefix}
```
{prompt}
```
"""
response = """\
{response_prefix}
```{language}
{_MAGIC_SPLITTER_}
```
"""


@hydra.main(config_path='../../configs', config_name='azr_minor_ppo_trainer', version_base=None)
def main(config):
    solve(config)
    
def solve(config):
    config.actor_rollout_ref.actor.optim = None
    ray.init(
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
        num_cpus=config.ray_init.num_cpus,
        _temp_dir='/tmp/ray/inoue'
    )
    
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=trust_remote_code)
    
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }
    resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
    mapping = {
            Role.ActorRollout: "global_pool",
        }
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
                cls=role_worker_mapping[Role.ActorRollout],
                config=config.actor_rollout_ref,
                role="actor_rollout",
            )
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
    wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
    if OmegaConf.select(config.trainer, "ray_wait_register_center_timeout") is not None:
        wg_kwargs["ray_wait_register_center_timeout"] = config.trainer.ray_wait_register_center_timeout
    if OmegaConf.select(config.trainer, "profile_steps") is not None:
        wg_kwargs["profile_steps"] = OmegaConf.select(config.trainer, "profile_steps")
        assert OmegaConf.select(config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
        wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(config.trainer, "worker_nsight_options"))
    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name="cuda", **wg_kwargs)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    rollout = all_wg['actor_rollout']
    rollout.init_model()
    if config.data.load_checkpoint:
        checkpoint_path = (Path(config.trainer.default_local_dir) / config.data.train_files.split('/')[-1].split('.')[0] / config.actor_rollout_ref.model.path.split('/')[-1] / config.reward_fn.extraction_type / f"global_step_{config.data.checkpoint_global_step}" / "actor").as_posix()
        rollout.load_checkpoint(checkpoint_path)
    print("Model loaded")
    executor = SandboxfusionExecutor(
        language = config.azr.language,
        use_china_mirror = False
    )
    print("Executor initialized")
    
    test_path = 'azr_minor/evaluation/data/' + config.data.test_file
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    
    test_ds = []
    len_data = len(test_data)
    for i, item in enumerate(test_data):
        prompt = item['prompt'].strip() + '\n' 
        """
        prompt = task_prompt.format(instruction_prefix=instruction_prefix[config.azr.evalperf_type].format(Language=Language[config.azr.language]), prompt=prompt)
        response_prompt = response.format(response_prefix=response_prefix[config.azr.evalperf_type].format(Language=Language[config.azr.language]), language=config.azr.language, _MAGIC_SPLITTER_=_MAGIC_SPLITTER_)
        test_data[i]['response_header'] = response_prompt.split(_MAGIC_SPLITTER_)[0]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_prompt},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]
        """
        prompt = instruction.format(language=config.azr.language, Language=Language[config.azr.language], comment_out=comment_out[config.azr.language], question=prompt)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt}
            ],
            tokenize=False,
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt,
                                                                         tokenizer=tokenizer,
                                                                         max_length=config.data.max_prompt_length,
                                                                         pad_token_id=tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation='error')

        problem_prompt = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
        }
        problem_prompt['position_ids'] = compute_position_id_with_mask(attention_mask)[0]
        test_ds.append(problem_prompt)
    if len_data % config.data.test_batch_size != 0:
        for _ in range(config.data.test_batch_size - len_data % config.data.test_batch_size):
            test_ds.append(test_ds[-1])

    test_ds = iter(DataLoader(
            dataset=test_ds,
            batch_size=config.data.test_batch_size,
            collate_fn=collate_fn,
            sampler=SequentialSampler(test_ds)
        ))
    num_correct = 0
    for i in range(len(test_ds)):
        batch_dict = next(test_ds)
        batch = DataProto.from_single_dict(batch_dict)
        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        gen_batch_output = rollout.generate_sequences(gen_batch)
        for j in range(len(gen_batch_output)):
            if i * config.data.test_batch_size + j >= len_data:
                break
            output = gen_batch_output[j]
            prompt_length = output.batch['prompts'].shape[-1]
            valid_response_length = output.batch['attention_mask'][prompt_length:].sum()
            valid_output = output.batch['responses'][:valid_response_length]
            output_text = tokenizer.decode(valid_output)
            print(output_text)
            if config.azr.evalperf_type == 'azr':
                code_snippet = extract_code(output_text.split("<|fim_middle|>")[0].split("<answer>")[-1].split("</answer>")[0], config.azr.language)
            else:
                code_snippet = extract_code(test_data[i * config.data.test_batch_size + j]['response_header'] + output_text, config.azr.language)
                #code_snippet = extract_code(test_data[i * config.data.test_batch_size + j]['prompt'].split('\n')[-2] + '\n' + output_text, config.azr.language)
            code_snippet =  code_snippet + '\n' + test_data[i * config.data.test_batch_size + j]['tests'].strip()
            if config.azr.language == 'racket' and "#lang racket" not in code_snippet:
                code_snippet = "#lang racket\n\n" + code_snippet
            print(code_snippet)
            _, status = executor.apply(code_snippet)
            num_correct += status == 'done'
            
    print(num_correct, len(test_data))
    print(f'accuracy: {num_correct / len(test_data) * 100:.2f}')

if __name__ == "__main__":
    main()