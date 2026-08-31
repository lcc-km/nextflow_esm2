#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { HPO_SEARCH } from './modules/hpo_search.nf'
include { TRAIN_FINAL } from './modules/train_final.nf'
include { PREDICT }     from './modules/predict.nf'
/**
 * 从 CSV 文件路径中提取数据集名称（去掉路径和 .csv 后缀）
 * 用于命名输出目录和 WandB run
 */


/**
 * 构建 WandB run 名称
 */
def build_wandb_run_name(dataset_name, model_name, use_lora) {
    def model_short = model_name.tokenize('/')[-1]
    def lora_tag = use_lora ? "lora" : "full"
    return "${params.wandb_run_prefix}-${model_short}-${lora_tag}-${dataset_name}"
}


workflow SAMPLE_PARSING {
    take: input_file
    main:
        ch_samples = channel.fromPath(input_file)
            .splitCsv(header:true, sep: ',') 
            .map { row ->
                def meta = [
                    id: row.id,
                    sample: row.sample,
                    info: row.info,
                ]
                def data = row.data
                [meta, file(data)]
            }
    emit: ch_samples
}


workflow {

    log.info """
    ================================================================
    🧬 ESM-2 Fine-Tuning & DMS Screening Pipeline 🧬
    ================================================================
      训练数据 (data_csv)     : ${params.input}
      输出保存路径             : ${params.output_dir}
      预训练模型              : ${params.model_name}
      微调模式 (LoRA)         : ${params.use_lora} (r=${params.lora_r}, alpha=${params.lora_alpha})
      Batch Size             : ${params.batch_size} (梯度累积: ${params.gradient_accumulation_steps})
      优化器                  :  ${params.optimizer}
      WandB Project          : ${params.wandb_project}
    ================================================================
    """

    // 1. 加载数据
    SAMPLE_PARSING(params.input)

    // 2. 阶段一：HPO 超参数搜索 (快速运行)
    // 预期输出: tuple(meta, data_csv, path("best_hpo_params.json"))
    if (!params.skip_hpo) {
            // 正常运行 HPO 搜索
            HPO_SEARCH(SAMPLE_PARSING.out.ch_samples)

            HPO_SEARCH.out.hpo_results.view { meta, data, json_file -> 
                "🔍 [HPO 完成] 数据集: ${meta.id} | 最优参数文件已生成: ${json_file}" 
            }
            
            ch_final_train_input = HPO_SEARCH.out.hpo_results
        } else {
            // 跳过 HPO，将 json_file 占位设为空
            ch_final_train_input = SAMPLE_PARSING.out.ch_samples.map { meta, data ->
                return [meta, data, []]
            }
        }

    // 3. 阶段二：最终完整训练 (加载 HPO 结果，给足 Epochs)
    TRAIN_FINAL(HPO_SEARCH.out.hpo_results)

    TRAIN_FINAL.out.model.view { meta, dir -> 
        "💾 [最终训练完成] 数据集: ${meta.id} | 模型路径: ${dir}" 
    }

    // 4. 预测环节
    ch_predict_input = TRAIN_FINAL.out.model
        .map { meta, model_dir ->
            def predict_csv = file("${model_dir}/test.csv")
            return [meta, model_dir, predict_csv]
        }
    
    PREDICT(ch_predict_input, TRAIN_FINAL.out.wandb_run_id)
}