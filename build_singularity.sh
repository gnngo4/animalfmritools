docker build -t animalfmritools .
docker tag animalfmritools localhost:5000/animalfmritools
docker push localhost:5000/animalfmritools
rm animalfmritools.simg
SINGULARITY_NOHTTPS=1 singularity build animalfmritools.simg docker://localhost:5000/animalfmritools
