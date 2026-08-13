.PHONY: demo train evaluate kafka-up kafka-down clean

# One command: bring up the whole system and play the scripted fault scenario.
demo:
	./run_demo.sh

# Train the classifier (simulates faults offline -> model/classifier.pkl).
train:
	python -m model.train

# Offline evaluation: precision/recall/F1, confusion matrix, false-alarm rate,
# and time-to-detect.
evaluate:
	python -m model.evaluate

kafka-up:
	docker compose up -d

kafka-down:
	docker compose down

# Remove generated artifacts and caches.
clean:
	rm -f model/classifier.pkl model/training_data.csv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
