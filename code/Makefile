docker_number = 1

clear:
	pkill -9 python

build_container:
	docker build -t shuchen_pytorch_$(docker_number) docker/

start_container:
	make build_container
	GPU=0 /gpfs01/bethge/home/swu/docker-deeplearning/agmb-docker run --shm-size 32G -v /gpfs01/bethge/data/:/gpfs01/bethge/data -d swu_pytorch_$(docker_number)


