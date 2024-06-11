## Setup

Open `Git Bash` and clone this repository: `https://github.com/DavidJCottrell/digit-predictor.git`.

Create and source a virtual environment in the repo directory:

`python -m venv ./venv`

`source ./venv/Scripts/activate`

Install Dependencies:

`pip install PIL tensorflow numpy watchdog`

## Run Predictions

Open `test_number.png` in paint, draw a number and save, then run `python predict.py`.

![alt text](https://github.com/DavidJCottrell/digit-predictor/blob/main/demo.png)
