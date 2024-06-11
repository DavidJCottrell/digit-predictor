## Setup

Open `Git Bash` and clone this repository: `https://github.com/DavidJCottrell/digit-predictor.git`.

Create and source a virtual environment in the repo directory:

`python -m venv ./venv`

`source ./venv/Scripts/activate`

Install Dependencies:

`pip install pillow tensorflow numpy watchdog`

## Run Predictions

run `python predict.py` and open `test_number.png` in paint - the number will be predicted each time you save the file.

![alt text](https://github.com/DavidJCottrell/digit-predictor/blob/main/demo.png)
