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

#define _GNU_SOURCE       /* necessario para sched_getcpu() no Linux           */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>          /* so existe quando compilado com -fopenmp           */
#else
/* Fallbacks para compilar SEM -fopenmp (vira serial). Para paralelismo de
   verdade, compile com:  mpicc -fopenmp -O2 nqueen_t4.c -o nqueens_t4        */
static inline int omp_get_thread_num(void)  { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return 1; }
#endif
#include <unistd.h>       /* gethostname()                                     */
#ifdef __linux__
#include <sched.h>        /* sched_getcpu(): em qual nucleo a thread esta      */
#endif

#define TAG_TAREFA    0   /* mestre -> escravo: bloco de prefixos (3 ints cada) */
#define TAG_RESULTADO 1   /* escravo -> mestre: 1 long long (contagem)          */
#define TAG_FIM       2   /* mestre -> escravo: encerrar                        */

#define MAX_CHUNK   4096  /* maximo de prefixos por mensagem                    */
#define THREADS_TRABALHO 8 /* threads OpenMP por escravo (fixo, hard-coded)     */

/* === GLOBAIS === */
int  tamanho_tabuleiro = 15;   /* N (definido por argv no main)           */
int  escravos_vivos    = 0;    /* controlado pelo mestre                  */
int  meu_rank          = 0;    /* rank MPI deste processo                 */
char meu_host[256]     = "?";  /* nome do computador (no) deste processo  */

/* === PROTOTIPOS === */
int  place(int board_local[], int row, int col);
void queen(int board_local[], int row, int n, long long *count);

/* Em qual nucleo logico esta thread esta rodando agora (-1 se indisponivel) */
static int nucleo_atual(void) {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

/* ============================================================
 *  DIAGNOSTICO: mostra em quais nucleos as threads OpenMP caem.
 *  Faz um pequeno trabalho antes de imprimir para forcar o SO a
 *  realmente agendar cada thread num nucleo. Se TODAS as threads
 *  aparecerem no mesmo nucleo -> binding errado (estao amontoadas).
 * ============================================================ */
void diagnostico_nucleos(void) {
    #pragma omp parallel num_threads(THREADS_TRABALHO)
    {
        /* trabalho ficticio so para a thread ser escalonada de fato */
        volatile double x = 0.0;
        for (int k = 0; k < 3000000; k++) x += k * 0.5;
        (void)x;

        int tid  = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        int core = nucleo_atual();

        #pragma omp critical
        {
            printf("[NUCLEOS] host=%-12s rank=%d  thread %d de %d  -> nucleo logico %d\n",
                   meu_host, meu_rank, tid, nthr, core);
            fflush(stdout);
        }
    }
}

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

    /* So no PRIMEIRO bloco processado: cada thread reporta uma vez em qual
       nucleo ela esta calculando de verdade. Evita poluir a saida nos blocos
       seguintes. */
    static int ja_reportou = 0;
    int reportar = !ja_reportou;
    if (reportar) ja_reportou = 1;

    #pragma omp parallel for schedule(dynamic) reduction(+:total) num_threads(THREADS_TRABALHO)
    for (int i = 0; i < numPrefixos; i++) {
        /* cada thread imprime uma unica vez, na primeira iteracao que pegar */
        if (reportar) {
            static int impresso[256] = {0};   /* indexado por id de thread */
            int tid = omp_get_thread_num();
            if (tid < 256 && !impresso[tid]) {
                impresso[tid] = 1;
                #pragma omp critical
                {
                    printf("[TRABALHO] host=%-12s rank=%d  thread %d  calculando no nucleo logico %d\n",
                           meu_host, meu_rank, tid, nucleo_atual());
                    fflush(stdout);
                }
            }
        }

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

    meu_rank = rank;
    gethostname(meu_host, sizeof(meu_host));

    if (argc > 1) tamanho_tabuleiro = atoi(argv[1]);

    if (numProcessos < 2) {
        if (rank == 0)
            fprintf(stderr, "Use pelo menos 2 processos (1 mestre + 1 escravo).\n");
        MPI_Finalize();
        return 1;
    }

    /* --- MAPA DE PROCESSOS: qual rank esta em qual computador (no) --- */
    char *todos_hosts = NULL;
    if (rank == 0)
        todos_hosts = malloc((size_t)numProcessos * sizeof(meu_host));
    MPI_Gather(meu_host, sizeof(meu_host), MPI_CHAR,
               todos_hosts, sizeof(meu_host), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("=== MAPA DE PROCESSOS MPI (threads OpenMP por escravo = %d) ===\n",
               THREADS_TRABALHO);
        for (int r = 0; r < numProcessos; r++)
            printf("  rank %d  ->  host %-12s  [%s]\n",
                   r, &todos_hosts[r * sizeof(meu_host)],
                   r == 0 ? "MESTRE (coordena, nao calcula)" : "ESCRAVO (calcula)");
        printf("=================================================================\n");
        fflush(stdout);
        free(todos_hosts);
    }

    /* --- DIAGNOSTICO DE NUCLEOS: cada escravo mostra onde suas threads caem --- */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank != 0)
        diagnostico_nucleos();
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    if (rank == 0)
        mestre(numProcessos);
    else
        escravo();

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Tempo total: %.4f s  (processos MPI = %d, threads/escravo = %d)\n",
               t1 - t0, numProcessos, THREADS_TRABALHO);

    MPI_Finalize();
    return 0;
}
