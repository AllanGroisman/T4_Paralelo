/* ============================================================
 * N-Queens HIBRIDO  ->  MPI (mestre-escravo) + OpenMP (workpool)
 *
 * MPI  : 1 processo pesado por no. Rank 0 = coordenador (mestre),
 *        demais ranks = trabalhadores (escravos). O mestre distribui
 *        as colunas fixas das linhas 0,1,2 como tarefas.
 *
 * OpenMP: dentro de cada escravo, uma pool de tabuleiros parciais e
 *        consumida por todas as threads via schedule(dynamic).
 *        Sem thread mestre -> controle pela fila de iteracoes (workpool).
 *
 * Compilar:  mpicc -fopenmp -O2 nqueens_hibrido.c -o nqueens
 * Executar:  export OMP_NUM_THREADS=<nucleos_por_no>
 *            mpirun -np 4 --map-by ppr:1:node ./nqueens 15
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define TAG_TAREFA    0   /* mestre -> escravo: 3 ints (colunas 0,1,2)   */
#define TAG_RESULTADO 1   /* escravo -> mestre: 1 long long (contagem)   */

/* === GLOBAIS === */
int tamanho_tabuleiro = 15;   /* N (definido por argv no main)           */
int escravos_vivos    = 0;    /* controlado pelo mestre                  */

/* === PROTOTIPOS === */
int  place(int board_local[], int row, int col);
void queen(int board_local[], int row, int n, long long *count);

/* ============================================================
 *  FUNCOES NQUEENS (nucleo recursivo)
 * ============================================================ */
int place(int board_local[], int row, int col) {
    for (int i = 0; i < row; i++) {
        if (board_local[i] == col)
            return 0;
        if (abs(board_local[i] - col) == abs(i - row))
            return 0;
    }
    return 1;
}

void queen(int board_local[], int row, int n, long long *count) {
    if (row == n) {
        (*count)++;
        return;
    }
    for (int col = 0; col < n; col++) {
        if (place(board_local, row, col)) {
            board_local[row] = col;
            queen(board_local, row + 1, n, count);
            board_local[row] = -1;
        }
    }
}

/* ============================================================
 *  GERACAO DA POOL (expansao sequencial barata ate a linha de corte)
 * ============================================================ */

/* Conta quantos tabuleiros parciais validos existem ate linha_corte */
void contarTarefas(int board[], int row, int linha_corte, long *total) {
    if (row == linha_corte) { (*total)++; return; }
    for (int col = 0; col < tamanho_tabuleiro; col++) {
        if (place(board, row, col)) {
            board[row] = col;
            contarTarefas(board, row + 1, linha_corte, total);
            board[row] = -1;
        }
    }
}

/* Preenche a pool (array plano) com os tabuleiros parciais validos */
void gerarTarefas(int board[], int row, int linha_corte, int *pool, long *idx) {
    if (row == linha_corte) {
        memcpy(&pool[(*idx) * tamanho_tabuleiro], board,
               sizeof(int) * tamanho_tabuleiro);
        (*idx)++;
        return;
    }
    for (int col = 0; col < tamanho_tabuleiro; col++) {
        if (place(board, row, col)) {
            board[row] = col;
            gerarTarefas(board, row + 1, linha_corte, pool, idx);
            board[row] = -1;
        }
    }
}

/* ============================================================
 *  FUNCAO ESCRAVO: trabalhar com WORKPOOL OpenMP (dynamic)
 * ============================================================ */
long long trabalhar(int colunasIniciais[3]) {
    int n = tamanho_tabuleiro;

    /* Linhas 0,1,2 vem do mestre. Expandir +algumas linhas gera muitas
       tarefas independentes -> melhor balanceamento no schedule dynamic. */
    int linha_corte = (n > 6) ? 6 : n - 1;

    int board_base[n];
    memset(board_base, -1, sizeof(int) * n);
    board_base[0] = colunasIniciais[0];
    board_base[1] = colunasIniciais[1];
    board_base[2] = colunasIniciais[2];

    /* (1) GERACAO DA POOL */
    long total = 0;
    contarTarefas(board_base, 3, linha_corte, &total);
    if (total == 0) return 0;   /* configuracao inicial ja invalida */

    int *pool = malloc((size_t)total * n * sizeof(int));
    if (!pool) { fprintf(stderr, "Erro malloc pool\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    long idx = 0;
    gerarTarefas(board_base, 3, linha_corte, pool, &idx);

    /* (2) WORKPOOL OpenMP: todas as threads consomem a pool.
       schedule(dynamic) = cada thread pega a proxima tarefa livre,
       sem mestre. Cada tarefa tem seu proprio board -> sem corrida.
       reduction soma as contagens parciais com seguranca. */
    long long solucoes_locais = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:solucoes_locais)
    for (long t = 0; t < total; t++) {
        int board_thread[n];
        memcpy(board_thread, &pool[(size_t)t * n], sizeof(int) * n);
        long long count = 0;
        queen(board_thread, linha_corte, n, &count);
        solucoes_locais += count;
    }

    free(pool);
    return solucoes_locais;
}

/* ============================================================
 *  FUNCOES MESTRE
 * ============================================================ */

/* Envia {-1,-1,-1} para o escravo encerrar */
void matarEscravo(int processoEscravo) {
    int sinal_de_morte[3] = {-1, -1, -1};
    MPI_Send(sinal_de_morte, 3, MPI_INT, processoEscravo, TAG_TAREFA, MPI_COMM_WORLD);
    escravos_vivos--;
}

/* Gera todas as colocacoes validas de 3 rainhas (linhas 0,1,2).
   Retorna array plano [numTarefas*3] e escreve numTarefas. */
int* gerarTodasTarefas(long *numTarefas) {
    int n = tamanho_tabuleiro;
    long cap = 1024, qtd = 0;
    int *tarefas = malloc((size_t)cap * 3 * sizeof(int));

    int b[3];
    for (int c0 = 0; c0 < n; c0++) {
        b[0] = c0;
        for (int c1 = 0; c1 < n; c1++) {
            if (c1 == c0 || abs(c1 - c0) == 1) continue;          /* linha 1 vs 0 */
            b[1] = c1;
            for (int c2 = 0; c2 < n; c2++) {
                /* valida linha 2 contra linhas 0 e 1 */
                if (c2 == c0 || c2 == c1) continue;
                if (abs(c2 - c0) == 2) continue;                  /* diagonal vs linha 0 */
                if (abs(c2 - c1) == 1) continue;                  /* diagonal vs linha 1 */
                b[2] = c2;

                if (qtd == cap) {
                    cap *= 2;
                    tarefas = realloc(tarefas, (size_t)cap * 3 * sizeof(int));
                }
                tarefas[qtd*3 + 0] = b[0];
                tarefas[qtd*3 + 1] = b[1];
                tarefas[qtd*3 + 2] = b[2];
                qtd++;
            }
        }
    }
    *numTarefas = qtd;
    return tarefas;
}

void mestre(int numProcessos) {
    long numTarefas = 0;
    int *tarefas = gerarTodasTarefas(&numTarefas);

    escravos_vivos = numProcessos - 1;
    long long totalSolucoes = 0;
    long proxima = 0;

    /* Distribui a primeira tarefa para cada escravo */
    for (int escravo = 1; escravo < numProcessos; escravo++) {
        if (proxima < numTarefas) {
            MPI_Send(&tarefas[proxima*3], 3, MPI_INT, escravo, TAG_TAREFA, MPI_COMM_WORLD);
            proxima++;
        } else {
            matarEscravo(escravo);   /* mais escravos que tarefas */
        }
    }

    /* Laco principal: recebe resultado e reabastece o escravo */
    while (escravos_vivos > 0) {
        long long resultado;
        MPI_Status status;
        MPI_Recv(&resultado, 1, MPI_LONG_LONG, MPI_ANY_SOURCE,
                 TAG_RESULTADO, MPI_COMM_WORLD, &status);
        totalSolucoes += resultado;

        int origem = status.MPI_SOURCE;
        if (proxima < numTarefas) {
            MPI_Send(&tarefas[proxima*3], 3, MPI_INT, origem, TAG_TAREFA, MPI_COMM_WORLD);
            proxima++;
        } else {
            matarEscravo(origem);
        }
    }

    free(tarefas);
    printf("N = %d  ->  Total de solucoes = %lld\n", tamanho_tabuleiro, totalSolucoes);
}

/* ============================================================
 *  LACO ESCRAVO
 * ============================================================ */
void escravo(void) {
    while (1) {
        int tarefa[3];
        MPI_Status status;
        MPI_Recv(tarefa, 3, MPI_INT, 0, TAG_TAREFA, MPI_COMM_WORLD, &status);

        if (tarefa[0] == -1 && tarefa[1] == -1 && tarefa[2] == -1)
            break;   /* sinal de morte */

        long long count = trabalhar(tarefa);
        MPI_Send(&count, 1, MPI_LONG_LONG, 0, TAG_RESULTADO, MPI_COMM_WORLD);
    }
}

/* ============================================================
 *  MAIN
 * ============================================================ */
int main(int argc, char **argv) {
    int rank, numProcessos;

    /* OpenMP dentro do MPI -> nivel de thread FUNNELED basta:
       so a thread principal faz chamadas MPI. */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessos);

    if (argc > 1) tamanho_tabuleiro = atoi(argv[1]);

    if (numProcessos < 2) {
        if (rank == 0)
            fprintf(stderr, "Use pelo menos 2 processos (1 mestre + 1 escravo).\n");
        MPI_Finalize();
        return 1;
    }

    double t0 = MPI_Wtime();

    if (rank == 0)
        mestre(numProcessos);
    else
        escravo();

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Tempo total: %.4f s  (processos MPI = %d, threads/no = %d)\n",
               t1 - t0, numProcessos, 16);

    MPI_Finalize();
    return 0;
}