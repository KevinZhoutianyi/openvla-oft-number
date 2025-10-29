"""
Optional: Create charts from JSON data if you have matplotlib installed
This is NOT required - you can create charts in any tool you prefer
or upload to Overleaf without charts (tables are already in the .tex file)
"""

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import json
    from pathlib import Path
    
    print("Matplotlib is available! Creating charts...")
    
    # Create figures directory
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # Load data
    with open('plot_data/comparison_data.json') as f:
        comp_data = json.load(f)
    
    with open('plot_data/improvement_data.json') as f:
        imp_data = json.load(f)
    
    with open('plot_data/per_dim_errors.json') as f:
        dim_data = json.load(f)
    
    # Chart 1: Loss Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['gray', 'green', 'blue', 'orange', 'red']
    
    ax1.bar(comp_data['loss_types'], comp_data['accuracies'], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Action Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Action Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(comp_data['loss_types'], comp_data['l1_losses'], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('L1 Loss', fontsize=12, fontweight='bold')
    ax2.set_title('L1 Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created loss_comparison.png")
    
    # Chart 2: Per-dimension errors
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(dim_data['dimensions'], dim_data['mean_errors'], 
           yerr=dim_data['std_errors'], capsize=5, 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Action Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Per-Dimension Action Prediction Errors (L1 Loss)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'per_dim_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created per_dim_errors.png")
    
    # Chart 3: Improvement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors_imp = ['green', 'blue', 'orange', 'red']
    
    ax1.bar(imp_data['loss_types'], imp_data['accuracy_improvement'], 
            color=colors_imp, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Improvement over Base Model', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax2.bar(imp_data['loss_types'], imp_data['l1_improvement'], 
            color=colors_imp, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('L1 Loss Reduction', fontsize=12, fontweight='bold')
    ax2.set_title('L1 Loss Improvement over Base Model', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'improvement_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created improvement_chart.png")
    
    print(f"\n✨ Charts created in {figures_dir.absolute()}/")
    print("\nYou can now upload these to Overleaf figures/ folder!")
    
except ImportError:
    print("=" * 80)
    print("Matplotlib not available - that's OK!")
    print("=" * 80)
    print("\nOptions:")
    print("1. Create charts manually in Excel/Google Sheets using plot_data/*.json")
    print("2. Use online tools like plot.ly or DataWrapper")
    print("3. Skip charts - the presentation has tables which work great!")
    print("4. Install matplotlib: pip3 install --user matplotlib")
    print("\nThe presentation will compile fine without charts.")
    print("=" * 80)

