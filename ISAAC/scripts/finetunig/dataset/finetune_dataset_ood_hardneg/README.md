# PPE Dataset With OOD Hard Negatives

This dataset copies the original PPE fine-tuning dataset and adds OOD images to the training split with empty YOLO label files.

Purpose: reduce YOLO hallucinations on semantic OOD objects by explicit hard-negative exposure.

Base dataset: `scripts/finetunig/dataset/finetune_dataset`
Added OOD negative images: `187`
Source mode: `hallucination_report`
