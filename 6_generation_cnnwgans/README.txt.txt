Generation of audios using a cnn-wgan model
 
This readme describes the process of generation of audio using the cnn-wgan model described in:
 
Park, S., Elhilali, M., Han, D. K., & Ko, H. (2020). Amphibian sounds generating network based on adversarial learning. IEEE Signal Processing Letters, 27, 640-644.
 
This implementation is based on the original code of the authors which can be found in:
 
https://github.com/tkddnr7671/SinusoidalGAN
 
For running the code it is recommended to create a particular virtual environment.
Requirements: tensorflow>=1.15.0
This code was run locally using CPU.
To run the code:
 
1. Create folders for each specie s1, s2, …,s9
2. Create a folder audio inside each folder and put the training data there.
3. Run the training and generation using the following commands locally
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s1
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s2
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s3
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s4
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s5
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s6
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s7
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s8
python cnn.py --max_epoch=10000 --batch_size=20 --mode=1 --batch_size_gen=20 --eps_stop=0.000001 --working_dir=s9
4. Results of the generation will be stored in a folder called wave_loc inside the folder of each specie

