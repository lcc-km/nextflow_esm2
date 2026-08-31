process PREDICT {
    tag "$meta.id"
    container "my-cuda-py313:v1"
    secret 'WANDB_API_KEY'
    
    // 将预测结果发布到指定的输出目录
    publishDir(
    path: { "${params.output_dir}/${meta.id}/predictions" },  
    mode: 'copy',       
    overwrite: true,       
    createDir: true        
    )

    input:
    tuple val(meta), path(model_dir), path(predict_csv)
    tuple val(meta), path(wandb_run_id)

    output:
    tuple val(meta), path("*_predictions.csv"), emit: predictions_file

    script:
    """
    predict.py \\
            --model_dir ${model_dir} \\
            --data_csv ${predict_csv} \\
            --output_csv ${meta.id}_predictions.csv \\
            --base_model_name ${params.model_name} \\
            --scaler_path ${model_dir}/scaler.joblib \\
            --batch_size ${params.batch_size} \\
            --max_length ${params.max_length} \\
            --seq_column ${params.seq_column} \\
            --wandb_project ${params.wandb_project} \\
            --wandb_run_id ${wandb_run_id}
    """
}
