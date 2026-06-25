/* Teste serial da MESMA logica do nqueen_t4.c (sem MPI/OpenMP)
   para verificar se a contagem total bate. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tamanho_tabuleiro = 14;

int place(int b[], int row, int col){
    for(int i=0;i<row;i++){
        if(b[i]==col) return 0;
        if(abs(b[i]-col)==abs(i-row)) return 0;
    }
    return 1;
}
void queen(int b[], int row, int n, long long *count){
    if(row==n){(*count)++;return;}
    for(int col=0;col<n;col++){
        if(place(b,row,col)){ b[row]=col; queen(b,row+1,n,count); b[row]=-1; }
    }
}
void contarTarefas(int b[],int row,int corte,long*t){
    if(row==corte){(*t)++;return;}
    for(int c=0;c<tamanho_tabuleiro;c++){ if(place(b,row,c)){b[row]=c;contarTarefas(b,row+1,corte,t);b[row]=-1;} }
}
void gerarTarefas(int b[],int row,int corte,int*pool,long*idx){
    if(row==corte){ memcpy(&pool[(*idx)*tamanho_tabuleiro],b,sizeof(int)*tamanho_tabuleiro); (*idx)++; return; }
    for(int c=0;c<tamanho_tabuleiro;c++){ if(place(b,row,c)){b[row]=c;gerarTarefas(b,row+1,corte,pool,idx);b[row]=-1;} }
}
long long trabalhar(int ci[3]){
    int n=tamanho_tabuleiro;
    int corte=(n>6)?6:n-1;
    int base[n]; memset(base,-1,sizeof(int)*n);
    base[0]=ci[0];base[1]=ci[1];base[2]=ci[2];
    long total=0; contarTarefas(base,3,corte,&total);
    if(total==0) return 0;
    int*pool=malloc((size_t)total*n*sizeof(int));
    long idx=0; gerarTarefas(base,3,corte,pool,&idx);
    long long sol=0;
    for(long t=0;t<total;t++){
        int bt[n]; memcpy(bt,&pool[(size_t)t*n],sizeof(int)*n);
        long long c=0; queen(bt,corte,n,&c); sol+=c;
    }
    free(pool); return sol;
}
int main(int argc,char**argv){
    if(argc>1) tamanho_tabuleiro=atoi(argv[1]);
    int n=tamanho_tabuleiro;
    long long total=0;
    int b[3];
    for(int c0=0;c0<n;c0++){ b[0]=c0;
      for(int c1=0;c1<n;c1++){ if(c1==c0||abs(c1-c0)==1)continue; b[1]=c1;
        for(int c2=0;c2<n;c2++){
          if(c2==c0||c2==c1)continue;
          if(abs(c2-c0)==2)continue;
          if(abs(c2-c1)==1)continue;
          b[2]=c2;
          int t[3]={b[0],b[1],b[2]};
          total+=trabalhar(t);
        }
      }
    }
    printf("N=%d total=%lld\n",n,total);
    return 0;
}
