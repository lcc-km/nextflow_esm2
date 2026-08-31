def build_wandb_run_name(dataset_name, model_name, use_lora) {
    def model_short = model_name.tokenize('/')[-1]
    def lora_tag = use_lora ? "lora" : "full"
    def prefix = params.wandb_run_prefix ?: "run"
    return "${prefix}-${model_short}-${lora_tag}-${dataset_name}"
}

process TRAIN {
    tag "$meta.id"
    
    container "my-gpu-app:v1.2"

    secret 'WANDB_API_KEY'

    publishDir(
        path: { "${params.output_dir}/${meta.id}" },  
        mode: 'copy',       
        overwrite: true,       
        createDir: true        
    )

    input:
    tuple val(meta), path(csv_file)

    output:
    tuple val(meta), path("model_weights"), emit: model
    tuple val(meta), path("model_weights/wandb_run_id.txt"), emit: wandb_run_id

    script:
    def dataset_name = meta.id
    def wandb_run = build_wandb_run_name(dataset_name, params.model_name, params.use_lora)

    """
    train.py \\
            --data_csv ${csv_file} \\
            --dataset_name ${dataset_name} \\
            --output_dir ./model_weights \\
            --model_name ${params.model_name} \\
            --cache_dir ${params.cache_dir} \\
            --num_labels ${params.num_labels} \\
            --freeze_layers ${params.freeze_layers} \\
            --max_length ${params.max_length} \\
            --test_size ${params.test_size} \\
            --random_state ${params.random_state} \\
            --batch_size ${params.batch_size} \\
            --gradient_accumulation_steps ${params.gradient_accumulation_steps} \\
            --epochs ${params.epochs} \\
            --learning_rate ${params.learning_rate} \\
            --weight_decay ${params.weight_decay} \\
            --optimizer ${params.optimizer} \\
            --scheduler ${params.scheduler} \\
            --eta_min ${params.eta_min} \\
            --max_grad_norm ${params.max_grad_norm} \\
            --patience ${params.patience} \\
            --min_delta ${params.min_delta} \\
            --use_lora ${params.use_lora} \\
            --lora_r ${params.lora_r} \\
            --lora_alpha ${params.lora_alpha} \\
            --lora_dropout ${params.lora_dropout} \\
            --target_modules "${params.target_modules}" \\
            --use_hpo ${params.use_hpo} \\
            --n_trials ${params.n_trials} \\
            --use_amp ${params.use_amp} \\
            --wandb_project ${params.wandb_project} \\
            --wandb_run_name ${wandb_run} \\
            --save_best_only ${params.save_best_only}
            
    chmod -R 777 ./model_weights
    """
}