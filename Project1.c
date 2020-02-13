#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ROOT 0
#define ERROR -1
#define INF 99999
#define BUFFERSIZE 9999999
//whether or not to benchmark
#define BENCHMARK 0
//the number of times to run in order to average
#define BENCHMARKSIZE 3

enum
{
    SEQTAG,
    SCATTERTAG,
    GATHERTAG,
    FOXTAG
};

//matrix of size n by n
typedef struct SquareMatrix
{
    int **data;
    int size;
} * SquareMatrix;

//creates a new square matrix of size "size"
//initializes all values with 0
//assumes positive size
SquareMatrix SquareMatrix_create(int size)
{
    SquareMatrix newSquareMatrix = (SquareMatrix)malloc(sizeof(struct SquareMatrix));
    //not enough memory
    if (newSquareMatrix == NULL)
        return NULL;

    //create empty matrix
    //allocate memory for the matrix
    int *matrix = (int *)malloc(sizeof(int) * size * size);
    //not enough memory
    if (matrix == NULL)
        return NULL;

    //allocate pointers to the rows
    int **data = (int **)malloc(sizeof(int *) * size);
    //not enough memory
    if (data == NULL)
        return NULL;

    int i;
    //set pointers to the right location
    for (i = 0; i < size; i++)
    {
        data[i] = &(matrix[i * size]);
    }

    //initialize values of matrix to 0
    int j;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            data[i][j] = 0;
        }
    }

    newSquareMatrix->size = size;
    newSquareMatrix->data = data;
    return newSquareMatrix;
}

//copys a square matrix and returns the copy
SquareMatrix SquareMatrix_copy(SquareMatrix matrix)
{
    int size = matrix->size;
    SquareMatrix newMatrix = SquareMatrix_create(size);
    int i, j;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            newMatrix->data[i][j] = matrix->data[i][j];
        }
    }
    return newMatrix;
}

//prints a matrix to stdout
void SquareMatrix_print(SquareMatrix matrix)
{
    int i, j;
    for (i = 0; i < matrix->size; i++)
    {
        //print space for all but the last one in a line
        for (j = 0; j < matrix->size - 1; j++)
        {
            printf("%d ", matrix->data[i][j]);
        }
        printf("%d\n", matrix->data[i][matrix->size - 1]);
    }
}

//returns a new matrix with all values equal to a input value
SquareMatrix SquareMatrix_make_all(SquareMatrix matrix, int value)
{
    int i, j;
    SquareMatrix newMatrix = SquareMatrix_create(matrix->size);
    for (i = 0; i < matrix->size; i++)
    {
        for (j = 0; j < matrix->size; j++)
        {
            newMatrix->data[i][j] = value;
        }
    }
    return newMatrix;
}

//frees the memory of a square matrix
void SquareMatrix_free(SquareMatrix matrix)
{
    free(matrix->data[0]);
    free(matrix->data);
    free(matrix);
}

//communicators of the fox algorithm
typedef struct Comms
{
    MPI_Comm grid;
    MPI_Comm row;
    MPI_Comm column;
} * Comms;

//creates new comms and returns them
//all processes must participate
Comms Comms_create(int numProcs)
{
    Comms newComms = (Comms)malloc(sizeof(struct Comms));

    //create grid communicator
    int dims[2] = {sqrt(numProcs), sqrt(numProcs)};
    int periods[2] = {1, 1};
    int reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &newComms->grid);

    //create row communicator
    int dimsRow[2] = {0, 1};
    MPI_Cart_sub(newComms->grid, dimsRow, &newComms->row);

    //create column communicator
    int dimsColumn[2] = {1, 0};
    MPI_Cart_sub(newComms->grid, dimsColumn, &newComms->column);
    return newComms;
}

//print the information about the comms to the stdout
void Comms_print(Comms comms)
{
    //rank in the world
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    //rank and size of the grid communicator
    int gridSize, gridRank;
    MPI_Comm_size(comms->grid, &gridSize);
    MPI_Comm_rank(comms->grid, &gridRank);

    //rank and size of the row communicator
    int rowSize, rowRank;
    MPI_Comm_size(comms->row, &rowSize);
    MPI_Comm_rank(comms->row, &rowRank);

    //rank and size of the column communicator
    int columnSize, columnRank;
    MPI_Comm_size(comms->column, &columnSize);
    MPI_Comm_rank(comms->column, &columnRank);

    printf("Process %d\n\tGrid: size %d rank %d\n\tRow: size %d rank %d\n\tColumn: size %d rank %d\n", worldRank, gridSize, gridRank, rowSize, rowRank, columnSize, columnRank);
}

//frees the memory of comms
void Comms_free(Comms comms)
{
    MPI_Comm_free(&comms->column);
    MPI_Comm_free(&comms->row);
    MPI_Comm_free(&comms->grid);
    free(comms);
}

//stores the ranks of the communicators for easy access
typedef struct Ranks
{
    int grid;
    int row;
    int column;
} * Ranks;

//creates the new ranks and returns them
Ranks Ranks_create(Comms comms)
{
    Ranks ranks = (Ranks)malloc(sizeof(struct Ranks));

    MPI_Comm_rank(comms->grid, &ranks->grid);
    MPI_Comm_rank(comms->row, &ranks->row);
    MPI_Comm_rank(comms->column, &ranks->column);

    return ranks;
}

//prints the information about the ranks to the stdout
void Ranks_print(Ranks ranks)
{
    printf("Grid %d Row %d Column %d\n", ranks->grid, ranks->row, ranks->column);
}

//frees the memory of ranks
void Ranks_free(Ranks ranks)
{
    free(ranks);
}

//used to hold and acces data for the fox algorithm
typedef struct FoxData
{
    //the matrices to multiply
    SquareMatrix subMatrix1;
    SquareMatrix subMatrix2;

    //communicators
    Comms comms;

    //rank in the communicators
    Ranks ranks;

    //data to relate original matrix with te sub matrices
    int numProcs;
    int subMatrixSize;
    int matrixSizeRatio;

    //ranks in the column from where to receive and send data
    int sourceRank;
    int destRank;
} * FoxData;

//creates the data for the fox algorithm and returns it
FoxData FoxData_create(SquareMatrix subMatrix1, SquareMatrix subMatrix2, int numProcs, int matrixSizeRatio)
{
    FoxData foxData = (FoxData)malloc(sizeof(struct FoxData));

    foxData->subMatrix1 = SquareMatrix_copy(subMatrix1);
    foxData->subMatrix2 = SquareMatrix_copy(subMatrix2);

    foxData->comms = Comms_create(numProcs);
    foxData->ranks = Ranks_create(foxData->comms);

    foxData->numProcs = numProcs;
    foxData->subMatrixSize = subMatrix1->size;
    foxData->matrixSizeRatio = matrixSizeRatio;

    //receive from the next row. If last row loop around
    foxData->sourceRank = (foxData->ranks->column + 1) % matrixSizeRatio;
    //send to the previous row. If first row loop around
    foxData->destRank = (foxData->ranks->column + matrixSizeRatio - 1) % matrixSizeRatio;

    return foxData;
}

//prints all the data in fox data to the stdout
void FoxData_print(FoxData foxData)
{
    printf("matrix1:\n");
    SquareMatrix_print(foxData->subMatrix1);
    printf("matrix2:\n");
    SquareMatrix_print(foxData->subMatrix2);
    printf("comms:\n");
    Comms_print(foxData->comms);
    printf("ranks:\n");
    Ranks_print(foxData->ranks);
    printf("numProcs: %d\n", foxData->numProcs);
    printf("subMatrixSize: %d\n", foxData->subMatrixSize);
    printf("matrixSizeRatio: %d\n", foxData->matrixSizeRatio);
    printf("sourceRank: %d\n", foxData->sourceRank);
    printf("destRank: %d\n", foxData->destRank);
}

//frees the memory of the fox data
void FoxData_free(FoxData foxData)
{
    SquareMatrix_free(foxData->subMatrix1);
    SquareMatrix_free(foxData->subMatrix2);
    Comms_free(foxData->comms);
    Ranks_free(foxData->ranks);
    free(foxData);
}

//reads input file from stdin
//returns a matrix with the data from the file
SquareMatrix read_input()
{
    //get matrix size
    int nNodes;
    scanf("%d", &nNodes);

    SquareMatrix matrix = SquareMatrix_create(nNodes);

    //read matrix values
    int i, j;
    for (i = 0; i < matrix->size; i++)
    {
        for (j = 0; j < matrix->size; j++)
        {
            scanf("%d", &matrix->data[i][j]);
        }
    }

    return matrix;
}

//returns the minimum of two values
int minimum(int v1, int v2)
{
    if (v1 > v2)
    {
        return v2;
    }
    return v1;
}

//returns a new matrix with values of 0 outside the diagonal turned into infinity
//infinity is a large constant, not actual infinity
SquareMatrix infinify(SquareMatrix matrix)
{
    int size = matrix->size;
    SquareMatrix outMatrix = SquareMatrix_copy(matrix);
    int row, column;
    for (row = 0; row < size; row++)
    {
        for (column = 0; column < size; column++)
        {
            if (row != column && matrix->data[row][column] == 0)
            {
                outMatrix->data[row][column] = INF;
            }
        }
    }
    return outMatrix;
}

//returns a new  matrix with values of infinity turned into zero
//infinity is a large constant, not actual infinity
SquareMatrix deinfinify(SquareMatrix matrix)
{
    int size = matrix->size;
    SquareMatrix outMatrix = SquareMatrix_copy(matrix);
    int row, column;
    for (row = 0; row < size; row++)
    {
        for (column = 0; column < size; column++)
        {
            if (matrix->data[row][column] == INF)
            {
                outMatrix->data[row][column] = 0;
            }
        }
    }
    return outMatrix;
}

//calculates the minPlus operation of two matrices
//assumes matrices of equal size
//returns matrix with solution
SquareMatrix minPlus(SquareMatrix matrix1, SquareMatrix matrix2)
{
    int size = matrix1->size;
    SquareMatrix outMatrix = SquareMatrix_make_all(SquareMatrix_create(matrix1->size), INF);

    int row, column, offset;
    int leftValue, rightValue;
    int min;
    for (row = 0; row < size; row++)
    {
        for (column = 0; column < size; column++)
        {
            for (offset = 0; offset < size; offset++)
            {
                leftValue = matrix1->data[row][offset];
                rightValue = matrix2->data[offset][column];

                outMatrix->data[row][column] = minimum(outMatrix->data[row][column], leftValue + rightValue);
            }
        }
    }
    return outMatrix;
}

//returns new matrix with every value being the minimum between the values of two matrices
SquareMatrix element_wise_min(SquareMatrix matrix1, SquareMatrix matrix2)
{
    SquareMatrix outMatrix = SquareMatrix_copy(matrix1);

    int row, column;
    int size = matrix1->size;
    for (row = 0; row < size; row++)
    {
        for (column = 0; column < size; column++)
        {
            outMatrix->data[row][column] = minimum(matrix1->data[row][column], matrix2->data[row][column]);
        }
    }
    return outMatrix;
}

//ends process and sends termination message to all other
void terminate()
{
    int errorMessage = ERROR;
    MPI_Bcast(&errorMessage, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Finalize();
    exit(0);
}

//validates parameters
//terminates program if parameters are invalid
void validate_parameters(SquareMatrix matrix, int numProcs)
{
    //verify that the number of processes is a perfect square
    double temp = sqrt((double)numProcs);
    int sideSize = temp;
    //numProcs is not a perfect square
    if (sideSize != temp)
    {
        fprintf(stderr, "ERROR: invalid number of processes(must be a perfect square)\n");
        SquareMatrix_free(matrix);
        terminate();
    }

    //read input failed
    if (matrix == NULL)
    {
        SquareMatrix_free(matrix);
        terminate();
    }
    //invalid number of processes for the size of the matrix
    else if (matrix->size % sideSize != 0)
    {
        fprintf(stderr, "ERROR: invalid number of processes for matrix size\n");
        SquareMatrix_free(matrix);
        terminate();
    }
}

//creates and returns the mpi type of a matrix
MPI_Datatype create_matrix_type(int size, int numProcs)
{
    MPI_Datatype MPIMatrix;
    MPI_Type_vector(size, size, size, MPI_INT, &MPIMatrix);
    MPI_Type_commit(&MPIMatrix);
    return MPIMatrix;
}

//scatters a matrix by all processes
//divides it into smaller square matrices
//only called by root
SquareMatrix scatter_matrix(SquareMatrix matrix, int numProcs)
{
    int matrixSizeRatio = sqrt(numProcs);
    int subMatrixSize = matrix->size / matrixSizeRatio;
    int stride = matrix->size;
    //mpi type of a square sub matrix of another matrix
    MPI_Datatype MPISubMatrix;
    MPI_Type_vector(subMatrixSize, subMatrixSize, stride, MPI_INT, &MPISubMatrix);
    MPI_Type_commit(&MPISubMatrix);

    MPI_Datatype MPIMatrix = create_matrix_type(subMatrixSize, numProcs);
    SquareMatrix mySubMatrix = SquareMatrix_create(subMatrixSize);
    int i, j;
    for (i = 0; i < matrixSizeRatio; i++)
    {
        for (j = 0; j < matrixSizeRatio; j++)
        {
            MPI_Bsend(matrix->data[i * subMatrixSize] + j * subMatrixSize, 1, MPISubMatrix, i * matrixSizeRatio + j, SCATTERTAG, MPI_COMM_WORLD);
        }
    }
    MPI_Recv(*mySubMatrix->data, 1, MPIMatrix, ROOT, SCATTERTAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Type_free(&MPISubMatrix);
    MPI_Type_free(&MPIMatrix);
    return mySubMatrix;
}

//each process print a matrix sequentially
//all processes must participate
void sequential_print(SquareMatrix matrix, int rank, int numProcs)
{
    //just print if only one process
    if (numProcs == 1)
    {
        fprintf(stderr, "Proc %d matrix:\n", rank);
        SquareMatrix_print(matrix);
        return;
    }
    int message = 1;
    //last process doesn't send a message to the next
    if (rank == numProcs - 1)
    {
        MPI_Recv(&message, 1, MPI_INT, rank - 1, SEQTAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stderr, "Proc %d matrix:\n", rank);
        SquareMatrix_print(matrix);
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
    //process 0 prints first
    if (rank == 0)
    {
        printf("Proc %d matrix:\n", rank);
        SquareMatrix_print(matrix);
        MPI_Bsend(&message, 1, MPI_INT, rank + 1, SEQTAG, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //other processes wait to receive a message before printing
    else
    {
        MPI_Recv(&message, 1, MPI_INT, rank - 1, SEQTAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Proc %d matrix:\n", rank);
        SquareMatrix_print(matrix);
        MPI_Bsend(&message, 1, MPI_INT, rank + 1, SEQTAG, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

//gather all sub matrices into a main matrix
//only called by root
//returns the gathered matrix
SquareMatrix gather_matrix(int size, int numProcs)
{
    int matrixSizeRatio = sqrt(numProcs);
    int subMatrixSize = size / matrixSizeRatio;
    int stride = size;
    //mpi type of a square sub matrix of another matrix
    MPI_Datatype MPISubMatrix;
    MPI_Type_vector(subMatrixSize, subMatrixSize, stride, MPI_INT, &MPISubMatrix);
    MPI_Type_commit(&MPISubMatrix);

    SquareMatrix gatheredMatrix = SquareMatrix_create(size);

    int i, j;
    for (i = 0; i < matrixSizeRatio; i++)
    {
        for (j = 0; j < matrixSizeRatio; j++)
        {
            MPI_Recv(gatheredMatrix->data[i * subMatrixSize] + j * subMatrixSize, 1, MPISubMatrix, i * matrixSizeRatio + j, GATHERTAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Type_free(&MPISubMatrix);
    return gatheredMatrix;
}

//fox algorithm
//returns a new square matrix with the solution
SquareMatrix fox(FoxData foxData)
{
    SquareMatrix tempMatrix = SquareMatrix_create(foxData->subMatrixSize);
    SquareMatrix outMatrix = SquareMatrix_make_all(SquareMatrix_create(foxData->subMatrixSize), INF);

    int step;
    int root;

    MPI_Datatype MPIMatrix = create_matrix_type(foxData->subMatrixSize, foxData->numProcs);

    for (step = 0; step < foxData->matrixSizeRatio; step++)
    {
        //chose root of each row
        root = (foxData->ranks->column + step) % foxData->matrixSizeRatio;

        //if root send matrix to all processes in same row
        if (root == foxData->ranks->row)
        {
            MPI_Bcast(*foxData->subMatrix1->data, 1, MPIMatrix, root, foxData->comms->row);
            outMatrix = element_wise_min(outMatrix, minPlus(foxData->subMatrix1, foxData->subMatrix2));
        }
        //otherwise receive matrix for m root
        else
        {
            MPI_Bcast(*tempMatrix->data, 1, MPIMatrix, root, foxData->comms->row);
            outMatrix = element_wise_min(outMatrix, minPlus(tempMatrix, foxData->subMatrix2));
        }
        //all processes send the matrix to the process above them
        MPI_Bsend(*foxData->subMatrix2->data, 1, MPIMatrix, foxData->destRank, FOXTAG, foxData->comms->column);
        MPI_Recv(*foxData->subMatrix2->data, 1, MPIMatrix, foxData->sourceRank, FOXTAG, foxData->comms->column, MPI_STATUS_IGNORE);
    }

    SquareMatrix_free(tempMatrix);
    MPI_Type_free(&MPIMatrix);

    return outMatrix;
}

//Calculates log2 of a number.
double Log2(int n)
{
    // log(n)/log(2) is log2.
    return log(n) / log(2);
}

//calculates the all pairs shortest path of a matrix
//also calculates the time if in benchmark mode
SquareMatrix all_pairs_shortest_path(SquareMatrix inMatrix, int numProcs)
{
    //variables for the fox algorithm
    int matrixSizeRatio = sqrt(numProcs);
    int step, steps = ceil(log2(inMatrix->size * matrixSizeRatio));
    SquareMatrix outMatrix = SquareMatrix_copy(inMatrix);
    FoxData foxData;

    //variables for the benchmark
    double start, end;
    double cumTime = 0;
    int i;

    for (i = 0; i < BENCHMARKSIZE; i++)
    {
        //get the start time
        if (BENCHMARK)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            start = MPI_Wtime();
        }

        //run the fox algorithm log2(n) times
        for (step = 0; step < steps; step++)
        {
            foxData = FoxData_create(outMatrix, outMatrix, numProcs, matrixSizeRatio);
            outMatrix = fox(foxData);
            if (step + 1 < steps)
            {
                FoxData_free(foxData);
            }
        }

        //get the end time and time delta
        if (BENCHMARK)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            //only get times and print if root
            if (foxData->ranks->grid == ROOT)
            {
                end = MPI_Wtime();
                printf("%d: %f seconds\n", i, end - start);
                cumTime += end - start;
            }
        }
        // if not in benchmark mode after one computation exit
        else
        {
            return outMatrix;
        }
    }
    //only print if root
    if (foxData->ranks->grid == ROOT)
    {
        printf("The computation took an average %f seconds in %d runs\n", cumTime / BENCHMARKSIZE, BENCHMARKSIZE);
    }
    return outMatrix;
}

int main(int argc, char *argv[])
{
    int numProcs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //allocate buffer to use Bsend
    int *localBuffer = (int *)malloc(sizeof(int) * BUFFERSIZE);
    MPI_Buffer_attach(localBuffer, BUFFERSIZE);
    if (rank == ROOT)
    {
        //reads matrix from stdin
        SquareMatrix inMatrix = read_input();

        //terminate if invalid
        validate_parameters(inMatrix, numProcs);

        //broadcast matrix size
        MPI_Bcast(&inMatrix->size, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        //divide matrix among all processes
        SquareMatrix matrix = scatter_matrix(infinify(inMatrix), numProcs);

        matrix = all_pairs_shortest_path(matrix, numProcs);

        //optional command for debug and visualization purposes
        //sequential_print(matrix, rank, numProcs);

        //gather matrix from all processes
        MPI_Datatype MPIMatrix = create_matrix_type(matrix->size, numProcs);
        MPI_Bsend(*matrix->data, 1, MPIMatrix, ROOT, GATHERTAG, MPI_COMM_WORLD);
        SquareMatrix outMatrix = deinfinify(gather_matrix(inMatrix->size, numProcs));

        SquareMatrix_free(inMatrix);
        SquareMatrix_free(matrix);
        MPI_Type_free(&MPIMatrix);

        //write the output matrix
        SquareMatrix_print(outMatrix);

        SquareMatrix_free(outMatrix);
    }
    else
    {
        int size;
        MPI_Bcast(&size, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
        //an error has ocurred
        if (size == ERROR)
        {
            MPI_Finalize();
            exit(0);
        }
        else
        {
            //receive sub matrix from root process
            MPI_Datatype MPIMatrix = create_matrix_type(size / sqrt(numProcs), numProcs);
            SquareMatrix matrix = SquareMatrix_create(size / sqrt(numProcs));
            MPI_Recv(*matrix->data, 1, MPIMatrix, ROOT, SCATTERTAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            matrix = all_pairs_shortest_path(matrix, numProcs);

            //optional command for debug and visualization purposes
            //sequential_print(matrix, rank, numProcs);

            //send matrix to root for it to gather
            MPI_Bsend(*matrix->data, 1, MPIMatrix, ROOT, GATHERTAG, MPI_COMM_WORLD);

            SquareMatrix_free(matrix);
            MPI_Type_free(&MPIMatrix);
        }
    }
    MPI_Finalize();
}
