#!/bin/bash

# Check if the system is Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Running on Linux"
    python -m venv venv
    # Activate the virtual environment
    source venv/bin/activate

    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt

    # Train the model
    echo "Running trainer.py..."
    python trainer.py

    # Run the Flask app
    echo "Starting the Flask app..."
    python app.py
else
    echo "This script is only intended to run on Linux systems."
    exit 1
fi
