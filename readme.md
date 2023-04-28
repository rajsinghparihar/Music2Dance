# setup:
```
git clone https://github.com/rajsinghparihar/Music2Dance.git
cd ./Music2Dance
python -m venv env
./env/Scripts/activate
pip install -r requirements.txt
```
# to run the code:
```
python generate_dance.py \
--songpath './audio_files/bizcochito.mp3' \
--songname 'bizcochito' \
--steps 100 \
--type "action" \
--visfolder './images/popping_28'
```
