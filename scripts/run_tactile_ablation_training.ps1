#!/usr/bin/env powershell
# Tactile Ablation Study Training Loop
# Runs 3 training runs per augmentation variant with different seeds

$variants = @(
    "no_aug",
    "noise_only",
    "brightness_only", 
    "contrast_only",
    "blur_only",
    "channel_scaling_only",
    "all"
)

$ablation_configs = @{
    "no_aug" = "tactile_ablation_baseline.yml"
    "noise_only" = "tactile_ablation_noise_only.yml"
    "brightness_only" = "tactile_ablation_brightness_only.yml"
    "contrast_only" = "tactile_ablation_contrast_only.yml"
    "blur_only" = "tactile_ablation_blur_only.yml"
    "channel_scaling_only" = "tactile_ablation_channel_scaling_only.yml"
    "all" = "tactile_ablation_all.yml"
}

$base_folder = "Data/Data_split"
$seeds = @(42, 43, 44)

foreach ($variant in $variants) {
    Write-Host "`n" + "="*80
    Write-Host "VARIANT: $variant"
    Write-Host "="*80
    
    $variant_folder = "$base_folder/train_augmented_tactile_$variant"
    $train_augmented = "$base_folder/train_augmented"
    
    if (-not (Test-Path $variant_folder)) {
        Write-Host "ERROR: $variant_folder not found!" -ForegroundColor Red
        Write-Host "Run: python scripts/run_tactile_ablation.py first"
        continue
    }
    
    # Move variant to train_augmented
    if (Test-Path $train_augmented) {
        Remove-Item -Recurse -Force $train_augmented
    }
    Copy-Item -Recurse $variant_folder $train_augmented
    Write-Host "Copied $variant to train_augmented" -ForegroundColor Green
    
    $config = $ablation_configs[$variant]
    
    # Run 3 training runs
    foreach ($i in 0..2) {
        $run_num = $i + 1
        Write-Host "`n  Run $run_num/3 for $variant..." -ForegroundColor Yellow
        
        $seed = $seeds[$i]
        Write-Host "  Seed: $seed"
        
        # Train
        python scripts/train.py --config "configs/$config"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR in training run $run_num" -ForegroundColor Red
            break
        }
    }
    
    # Move back
    Copy-Item -Recurse $train_augmented "$variant_folder`_backup"
    Remove-Item -Recurse -Force $train_augmented
    Write-Host "Cleaned up $variant" -ForegroundColor Green
}

Write-Host "`n" + "="*80
Write-Host "ABLATION STUDY COMPLETE!"
Write-Host "="*80
Write-Host "Results are in: runs/tensorboard/"
Write-Host "Analyze with: python scripts/analyze_ablation.py"
