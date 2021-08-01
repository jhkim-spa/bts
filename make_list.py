import os
import glob


folders = os.listdir('./data/custom_data/depth')

f1 = open('train_list.txt', 'w')
f2 = open('val_list.txt', 'w')
f3 = open('test_list.txt', 'w')

for folder in folders:
	if folder not in ['test', 'validation']:
		filenames = glob.glob(f'data/custom_data/depth/{folder}/*.png')
		for filename in filenames:
			filename = filename.split('/')[-1]
			line = f'rgb/{folder}/{filename} depth/{folder}/{filename} 388\n'
			f1.write(line)
	elif folder in ['validation']:
		fds = glob.glob('./data/custom_data/depth/validation/*')
		for fd in fds:
			filenames = glob.glob(os.path.join(fd, '*.png'))
			for filename in filenames:
				fd = fd.split('/')[-1]
				filename = filename.split('/')[-1]
				line = f'rgb/{folder}/{fd}/{filename} depth/{folder}/{fd}/{filename} 388\n'
				f2.write(line)
	elif folder in ['test']:
		fds = glob.glob('./data/custom_data/depth/test/*')
		for fd in fds:
			filenames = glob.glob(os.path.join(fd, '*.png'))
			for filename in filenames:
				fd = fd.split('/')[-1]
				filename = filename.split('/')[-1]
				line = f'rgb/{folder}/{fd}/{filename} depth/{folder}/{fd}/{filename} 388\n'
				f3.write(line)
