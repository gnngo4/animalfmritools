docker run --rm -it \
	-v $PWD:/opt/animalfmritools \
	-v /mnt/WD10TB/datasets/rest:/data \
	-v /mnt/WD10TB/datasets/templates:/templates \
	-p 9889:9889 --expose 9889 \
	--entrypoint=jupyter animalfmritools lab \
	--port=9889 --no-browser --ip=0.0.0.0 \
	--notebook-dir=/opt/animalfmritools/notebooks --allow-root
