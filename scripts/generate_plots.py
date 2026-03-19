import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Move to project root
os.chdir(Path(__file__).resolve().parents[1])

MODELS = [
    "checkpoints/ppo.pt",
    "checkpoints/ppo_phase1.pt",
    "checkpoints/ppo_phase2.pt"
]

MODEL_LABELS = [
    "PPO (Base)",
    "PPO (Phase 1)",
    "PPO (Phase 2)"
]

OPPONENTS = ["random", "space_greedy"]

def evaluate_model(model_path, opponent, episodes=100, grid=20):
    cmd = [
        "python", "scripts/eval_ppo.py",
        "--model", model_path,
        "--opponent", opponent,
        "--episodes", str(episodes),
        "--grid", str(grid)
    ]
    
    print(f"Evaluating {model_path} vs {opponent}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output, e.g., episodes=100 wins=17 losses=58 draws=25 win_rate=0.170 avg_len=71.0
    output = result.stdout
    wins_match = re.search(r"wins=(\d+)", output)
    losses_match = re.search(r"losses=(\d+)", output)
    draws_match = re.search(r"draws=(\d+)", output)
    
    if wins_match and losses_match and draws_match:
        wins = int(wins_match.group(1))
        losses = int(losses_match.group(1))
        draws = int(draws_match.group(1))
        return wins, draws, losses
    else:
        print(f"Failed to parse output for {model_path} vs {opponent}:\n{output}\nError:\n{result.stderr}")
        return 0, 0, episodes

def main():
    os.makedirs("docs/assets", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('PPO Win Rate Progression', fontsize=16, fontweight='bold')
    
    colors = ['#2ca02c', '#ff7f0e', '#d62728'] # Green (Win), Orange (Draw), Red (Loss)
    
    for idx, opponent in enumerate(OPPONENTS):
        wins_list, draws_list, losses_list = [], [], []
        
        for model in MODELS:
            w, d, l = evaluate_model(model, opponent, episodes=100)
            total = w + d + l
            if total == 0: total = 1
            wins_list.append(w / total * 100)
            draws_list.append(d / total * 100)
            losses_list.append(l / total * 100)
            
        ax = axes[idx]
        ind = np.arange(len(MODEL_LABELS))
        width = 0.6
        
        p1 = ax.bar(ind, wins_list, width, color=colors[0], label='Wins')
        p2 = ax.bar(ind, draws_list, width, bottom=wins_list, color=colors[1], label='Draws')
        
        bottom_for_loss = [w + d for w, d in zip(wins_list, draws_list)]
        p3 = ax.bar(ind, losses_list, width, bottom=bottom_for_loss, color=colors[2], label='Losses')
        
        ax.set_title(f"vs '{opponent}'", fontweight='bold')
        ax.set_xticks(ind)
        ax.set_xticklabels(MODEL_LABELS, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Aesthetic improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        if idx == 1:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = "docs/assets/ppo_win_rate_progression.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
