/* ============================================================
 * N-Queens HIBRIDO  ->  MPI (mestre-escravo) + OpenMP (workpool)
 *
 * MPI   : Rank 0 = coordenador (mestre). Demais ranks = escravos.
 *         O mestre gera todos os prefixos validos das linhas 0,1,2 e os
 *         distribui em BLOCOS (chunks), reabastecendo cada escravo
 *         conforme ele termina (balanceamento dinamico no nivel MPI).
 *
 * OpenMP: cada escravo recebe um bloco de prefixos e abre UMA UNICA
 *         regiao paralela (`omp parallel for schedule(dynamic)`) sobre
 *         o bloco inteiro. Assim o custo de fork/join das threads e
 *         amortizado por muitos prefixos, e o `dynamic` equilibra
 *         subarvores de tamanhos diferentes. Sem malloc no caminho
 *         quente, sem pool intermediaria.
 *
 * IMPORTANTE (desempenho):
 *   - Em maquina com 8 nucleos FISICOS (16 logicos c/ HyperThreading),
 *     dimensione  (escravos que calculam) * OMP_NUM_THREADS ~= 8.
 *     Ex.: -np 2  +  OMP_NUM_THREADS=8   (1 mestre ocioso + 1 escravo).
 *   - NAO deixe o MPI prender cada rank a 1 core, senao as 8 threads
 *     OpenMP brigam por 1 nucleo. Use binding solto ou reserve cores:
 *       mpirun --bind-to none -np 2 ./nqueens 15
 *       mpirun --map-by node:PE=8 --bind-to core -np 2 ./nqueens 15
 *     e, opcionalmente:  export OMP_PROC_BIND=spread OMP_PLACES=cores
 *   - HyperThreading nao dobra desempenho em codigo CPU-bound: meca a
 *     eficiencia contra 8 (fisicos), nao 16.
 *
 * Compilar:  mpicc -fopenmp -O2 nqueen_t4.c -o nqueens
 * Executar:  export OMP_NUM_THREADS=8
 *            mpirun --bind-to none -np 2 ./nqueens 15
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define TAG_TAREFA    0   /* mestre -> escravo: bloco de prefixos (3 ints cada) */
#define TAG_RESULTADO 1   /* escravo -> mestre: 1 long long (contagem)          */
#define TAG_FIM       2   /* mestre -> escravo: encerrar                        */

#define MAX_CHUNK   4096  /* maximo de prefixos por mensagem                    */

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
 *  FUNCAO ESCRAVO: processa um BLOCO de prefixos com UMA regiao paralela
 *
 *  prefixos : array plano [numPrefixos*3] com as colunas das linhas 0,1,2.
 *  Uma unica regiao `omp parallel for schedule(dynamic)`:
 *    - fork/join pago uma vez por bloco (e nao uma vez por tarefa);
 *    - cada iteracao tem seu proprio tabuleiro local (sem condicao de corrida);
 *    - `dynamic` equilibra subarvores de tamanhos diferentes;
 *    - `reduction` soma as contagens parciais com seguranca.
 * ============================================================ */
long long trabalhar(const int *prefixos, int numPrefixos) {
    int n = tamanho_tabuleiro;
    long long total = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:total)
    for (int i = 0; i < numPrefixos; i++) {
        int board[n];
        memset(board, -1, sizeof(int) * n);
        board[0] = prefixos[i * 3 + 0];
        board[1] = prefixos[i * 3 + 1];
        board[2] = prefixos[i * 3 + 2];

        long long count = 0;
        queen(board, 3, n, &count);   /* explora a subarvore a partir da linha 3 */
        total += count;
    }

    return total;
}

/* ============================================================
 *  FUNCOES MESTRE
 * ============================================================ */

/* Manda o escravo encerrar (mensagem vazia com TAG_FIM) */
void matarEscravo(int processoEscravo) {
    MPI_Send(NULL, 0, MPI_INT, processoEscravo, TAG_FIM, MPI_COMM_WORLD);
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

/* Envia ate `chunk` prefixos a partir de `*proxima`. Retorna quantos enviou. */
static int enviarBloco(const int *tarefas, long numTarefas, long *proxima,
                       int chunk, int destino) {
    long restantes = numTarefas - *proxima;
    if (restantes <= 0) return 0;

    int envia = (restantes < chunk) ? (int)restantes : chunk;
    MPI_Send(&tarefas[(*proxima) * 3], envia * 3, MPI_INT,
             destino, TAG_TAREFA, MPI_COMM_WORLD);
    *proxima += envia;
    return envia;
}

void mestre(int numProcessos) {
    long numTarefas = 0;
    int *tarefas = gerarTodasTarefas(&numTarefas);

    escravos_vivos = numProcessos - 1;
    long long totalSolucoes = 0;
    long proxima = 0;

    /* Tamanho do bloco: pequeno o bastante para dar varios reabastecimentos
       por escravo (balanceamento MPI), grande o bastante para alimentar as
       threads OpenMP. ~8 reabastecimentos por escravo, limitado a MAX_CHUNK. */
    int chunk = (int)(numTarefas / ((long)escravos_vivos * 8));
    if (chunk < 1)         chunk = 1;
    if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;

    printf("Mestre: tabuleiro %dx%d, %d processos MPI, %d prefixos, chunk=%d\n",
           tamanho_tabuleiro, tamanho_tabuleiro, numProcessos,
           (int)numTarefas, chunk);

    /* Rajada inicial: um bloco para cada escravo */
    for (int escravo = 1; escravo < numProcessos; escravo++) {
        if (enviarBloco(tarefas, numTarefas, &proxima, chunk, escravo) == 0)
            matarEscravo(escravo);   /* mais escravos que tarefas */
    }

    /* Laco principal: recebe resultado e reabastece o escravo */
    while (escravos_vivos > 0) {
        long long resultado;
        MPI_Status status;
        MPI_Recv(&resultado, 1, MPI_LONG_LONG, MPI_ANY_SOURCE,
                 TAG_RESULTADO, MPI_COMM_WORLD, &status);
        totalSolucoes += resultado;

        int origem = status.MPI_SOURCE;
        if (enviarBloco(tarefas, numTarefas, &proxima, chunk, origem) == 0)
            matarEscravo(origem);
    }

    free(tarefas);
    printf("N = %d  ->  Total de solucoes = %lld\n", tamanho_tabuleiro, totalSolucoes);
}

/* ============================================================
 *  LACO ESCRAVO
 * ============================================================ */
void escravo(void) {
    int *buf = malloc((size_t)MAX_CHUNK * 3 * sizeof(int));
    if (!buf) { fprintf(stderr, "Erro malloc buffer escravo\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    while (1) {
        MPI_Status status;
        MPI_Recv(buf, MAX_CHUNK * 3, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_FIM)
            break;   /* sinal de encerramento */

        int recebidos = 0;
        MPI_Get_count(&status, MPI_INT, &recebidos);
        int numPrefixos = recebidos / 3;

        long long count = trabalhar(buf, numPrefixos);
        MPI_Send(&count, 1, MPI_LONG_LONG, 0, TAG_RESULTADO, MPI_COMM_WORLD);
    }

    free(buf);
}

/* ============================================================
 *  MAIN
 * ============================================================ */
int main(int argc, char **argv) {
    int rank, numProcessos;

    /* OpenMP dentro do MPI -> nivel FUNNELED basta: so a thread principal
       faz chamadas MPI (as chamadas MPI ficam fora das regioes paralelas). */
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
        printf("Tempo total: %.4f s  (processos MPI = %d, threads/escravo = %d)\n",
               t1 - t0, numProcessos, 8);

    MPI_Finalize();
    return 0;
}
