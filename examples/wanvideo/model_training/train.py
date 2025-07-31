import torch, os, json, argparse
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = []
        tokenizer_config = ModelConfig(path="./models/VACE1.3/google/umt5-xxl")
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, tokenizer_config=tokenizer_config)
        print(f"Loaded VACE MODEL FINALLY model configs.")
        #self.pipe.to("cuda")
        print(f"Loaded VACE MODEL FINALLY to {self.pipe.device}.")
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega) # type: ignore
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    
    # Initialize wandb if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_rank": args.lora_rank,
                "num_frames": args.num_frames,
                "dataset_base_path": args.dataset_base_path,
                "trainable_models": args.trainable_models,
                "lora_base_model": args.lora_base_model,
                "lora_target_modules": args.lora_target_modules,
                "save_every_n_epochs": args.save_every_n_epochs,
                "validate_every_n_epochs": args.validate_every_n_epochs,
                "validation_prompt": args.validation_prompt,
            }
        )
        print(f"Wandb initialized. Project: {args.wandb_project}, Run: {wandb.run.name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb logging requested but wandb is not installed. Install with: pip install wandb")
    
    dataset = VideoDataset(args=args)
    
    # Create validation dataset if validation path is provided
    validation_dataset = None
    if args.validate_every_n_epochs and args.validation_dataset_base_path:
        print(f"Loading validation dataset from: {args.validation_dataset_base_path}")
        
        # Create validation args based on training args but with validation paths
        validation_args = argparse.Namespace(**vars(args))
        validation_args.dataset_base_path = args.validation_dataset_base_path
        validation_args.dataset_metadata_path = args.validation_dataset_metadata_path
        
        try:
            validation_dataset = VideoDataset(args=validation_args)
            print(f"Validation dataset loaded with {len(validation_dataset)} samples")
        except Exception as e:
            print(f"Warning: Failed to load validation dataset: {e}")
            print("Validation will use prompt-based generation instead")
            validation_dataset = None
    elif args.validate_every_n_epochs:
        print("Validation enabled but no validation dataset path provided. Using prompt-based validation.")
    
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    )
    
    # Prepare validation config
    validation_config = {
        'prompt': args.validation_prompt,
        'negative_prompt': args.validation_negative_prompt,
        'num_frames': args.validation_num_frames,
        'height': args.validation_height,
        'width': args.validation_width,
        'seed': args.validation_seed,
    }
    
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        use_wandb=args.use_wandb,
        save_every_n_epochs=args.save_every_n_epochs,
        validate_every_n_epochs=args.validate_every_n_epochs,
        validation_config=validation_config,
        validation_dataset=validation_dataset
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    print(f"Starting training with {len(dataset)} samples.")
    if args.validate_every_n_epochs:
        print(f"Validation will run every {args.validate_every_n_epochs} epochs")
    print(f"Checkpoints will be saved every {args.save_every_n_epochs} epochs")
    
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
