import matplotlib.pyplot as plt

# Checkpoints and corresponding mean perplexity values for new data
checkpoints = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
mean_perplexities = [7.178369957447052, 7.265611580848693, 6.720559018611908, 6.825776010036469, 
                     6.606883071422577, 6.293017171382904, 6.3809881882667545, 6.397646416187286, 
                     6.476371746063233, 6.101068610191345]

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
plt.savefig('learning_curve_tai.png')
# Show plot
plt.show()
