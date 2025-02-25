import matplotlib.pyplot as plt

# Checkpoints and corresponding mean perplexity values
checkpoints = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
mean_perplexities = [20.17345979356766, 19.74833458662033, 18.071825941562654, 18.333080658435822, 
                     16.896250476837157, 16.363145005226134, 16.132947598457335, 16.283352972984314, 
                     15.76722992324829, 15.785838223457336]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(checkpoints, mean_perplexities, marker='o', color='blue', linestyle='-', linewidth=1.5)
for i, txt in enumerate(mean_perplexities):
    plt.annotate(f"{txt:.2f}", (checkpoints[i], mean_perplexities[i]), textcoords="offset points", xytext=(0,5), ha='center')

# Labels and title
plt.xlabel("Checkpoint")
plt.ylabel("Mean Perplexity")
plt.title("Mean Perplexity across Checkpoints")
plt.grid(True)
plt.savefig('learning_curve.png')

# Show plot
plt.show()
