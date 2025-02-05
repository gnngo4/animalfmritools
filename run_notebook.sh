docker run --rm -it \
	-v $PWD:/opt/animalfmritools \
	-v /mnt/WD10TB/datasets/rest:/data \
	-v /mnt/WD10TB/datasets/templates:/templates \
	-p 9890:9890 --expose 9890 \
	--entrypoint=jupyter animalfmritools_palm lab \
	--port=9890 --no-browser --ip=0.0.0.0 \
	--notebook-dir=/opt/animalfmritools/notebooks --allow-root
