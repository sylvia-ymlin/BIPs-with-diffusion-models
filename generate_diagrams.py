
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, facecolor='white', edgecolor='black'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12, fontweight='bold')
    return x, y, width, height

def draw_arrow(ax, start, end, label=None):
    # start/end are (x, y)
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2))
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.05, label, ha='center', va='bottom', fontsize=10)

def generate_graphical_abstract():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Nodes
    # Prior
    draw_box(ax, 1, 4, 2.5, 1, "Prior:\nDiffusion p(x)", facecolor='#e6f3ff', edgecolor='#0066cc')
    # Data
    draw_box(ax, 1, 1, 2.5, 1, "Data:\ny = Ax + n", facecolor='#e6f3ff', edgecolor='#0066cc')
    
    # Sampler (Central)
    draw_box(ax, 5, 2.5, 3, 1.5, "Twisted SMC\nSampler", facecolor='#ffcccc', edgecolor='#cc0000')

    # Outputs
    draw_box(ax, 9.5, 4, 2, 1, "Posterior\nMean", facecolor='#e6ffe6', edgecolor='#006600')
    draw_box(ax, 9.5, 1, 2, 1, "Uncertainty\nMap", facecolor='#e6ffe6', edgecolor='#006600')

    # Arrows
    draw_arrow(ax, (3.6, 4.5), (5.0, 3.8), "Score")
    draw_arrow(ax, (3.6, 1.5), (5.0, 2.7), "Likelihood")
    
    draw_arrow(ax, (8.1, 3.8), (9.4, 4.5), "Particles")
    draw_arrow(ax, (8.1, 2.7), (9.4, 1.5), "Variance")
    
    plt.title("Graphical Abstract: Bayesian Inverse Problems via Diffusion Priors", fontsize=16, pad=20)
    plt.savefig("results/graphical_abstract.png", bbox_inches='tight', dpi=300)
    plt.close()

def generate_uml():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # UML Class Boxes
    # Solver
    draw_box(ax, 4, 6, 2, 1.5, "DiffusionSolver\n\n+ sample()", facecolor='white')

    # Strategy Interface
    draw_box(ax, 1, 3, 2.5, 1.5, "<<Interface>>\nSamplingStrategy\n\n+ update_step()", facecolor='#fff2cc')
    
    # Physics Interface
    draw_box(ax, 6.5, 3, 2.5, 1.5, "<<Interface>>\nPhysicsOperator\n\n+ forward()\n+ distance()", facecolor='#fff2cc')

    # Implementations (Abstract representation)
    draw_box(ax, 0.5, 0.5, 1.5, 1, "TwistedSMC", facecolor='#f0f0f0')
    draw_box(ax, 2.5, 0.5, 1.5, 1, "DPS", facecolor='#f0f0f0')

    draw_box(ax, 6.0, 0.5, 1.5, 1, "SuperRes", facecolor='#f0f0f0')
    draw_box(ax, 8.0, 0.5, 1.5, 1, "MRI", facecolor='#f0f0f0')

    # Connectors
    # Solver uses Strategy
    draw_arrow(ax, (4.5, 6), (2.25, 4.6), "uses")
    # Solver uses Physics
    draw_arrow(ax, (5.5, 6), (7.75, 4.6), "uses")
    
    # Inheritance
    ax.annotate("", xy=(1.75, 3.0), xytext=(1.25, 1.6), arrowprops=dict(arrowstyle="-|>", lw=1.5, ls='dashed'))
    ax.annotate("", xy=(1.75, 3.0), xytext=(3.25, 1.6), arrowprops=dict(arrowstyle="-|>", lw=1.5, ls='dashed'))
    
    ax.annotate("", xy=(7.75, 3.0), xytext=(6.75, 1.6), arrowprops=dict(arrowstyle="-|>", lw=1.5, ls='dashed'))
    ax.annotate("", xy=(7.75, 3.0), xytext=(8.75, 1.6), arrowprops=dict(arrowstyle="-|>", lw=1.5, ls='dashed'))

    plt.title("Modular Architecture (Strategy Pattern)", fontsize=16, pad=20)
    plt.savefig("results/architecture_uml.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_graphical_abstract()
    generate_uml()
    print("Generated diagrams in results/")
