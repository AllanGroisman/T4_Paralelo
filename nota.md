ladcomp -env mpicc nqueen.c -o nqueen_exec


RODAR NOVAMENTE T2:

FORTE:
Sem HT
srun -N 1 -n 2 ./nqueen_exec 14 - 30s (sequencial - 1 coordenador + 1 trabalhador) 2 nucleos

srun -N 4 -n 8 ./nqueen_exec 14 -  4,35   (1 coordenador + 7 trabalhador) 8 nucleos
srun -N 4 -n 16 ./nqueen_exec 14 - 2,15    (1 coordenador + 15 trabalhador) 16 nucleos
srun -N 3 -n 24
srun -N 4 -n 32 ./nqueen_exec 14 - 1,28    (1 coordenador + 31 trabalhador) 32 nucleos

Com HT
srun -N 4 -n 64 ./nqueen_exec 14 -  1,02  (1 coordenador + 63 trabalhador) 64 nucleos




RODAR T4:
FORTE
Sem HT
srun -N 1 -n 2 ./nqueen_exec 14 -          (1 coordenador + 1 gerenciador + 8 trabalhores)  8
srun -N 2 -n 3 ./nqueen_exec 14 -          (1 coordenador + 2 gerenciadores + 8 trabalhadores) 16
srun -N 3 -n 4 ./nqueen_exec 14 -          (1 coordenador + 3 gerenciadores + 8 trabalhadores) 24
srun -N 4 -n 5 ./nqueen_exec 14 -          (1 coordenador + 4 gerenciadores + 8 trabalhadores) 32
srun -N 4 -n 5 ./nqueen_exec 14 -          (1 coordenador + 4 gerenciadores + 16 trabalhadores) 64


