package CS4200;

//  Array[Col][Row]

/*


0 0 0 6 0 0 
0 0 0 0 0 0
5 0 4 1 0 0
0 0 0 0 0 3
0 0 2 0 0 0
0 0 0 0 0 0

[row][col]

RIGHT TO BOT LEFT

1: [2][3]	Diag: [][]

2: [4][2]	Diag: [][]

3: [3][5]	Diag: [][]

4: [2][2]	Diag: [][] 

Diagonal={(x,y)∣x,y∈R,if x>y then (∣x−y∣,0), else if y>x then (0,∣x−y∣)}


LEFT TO BOT RIGHT 

1: [2][3]	Diag: [][]

2: [4][2]	Diag: [][]

3: [3][5]	Diag: [][]

4: [2][2]	Diag: [][]

5: [2][0]	Diag: [][]

6: [0][3]	Diag: [][]

 */

public class Chessboard {
    private int[][] chessboard;
    private int chessboardWidth;
    private int chessboardHeight;


    
    public Chessboard(int dimX, int dimY)
    {
        this.chessboardWidth = dimX;
        this.chessboardHeight = dimY;
        this.chessboard = new int[dimX][dimY];
    }

    public void printChessboardOnConsole()
    {
        System.out.println();

        for(int row = 0; row < chessboard[0].length; row++)
        {
            for (int col = 0; col < chessboard.length; col++)
            {
                if(isQueen(col, row))
                {
                    System.out.printf("[X]  ");
                }else{
                    System.out.printf("[ ]  ");
                }
            }
            System.out.println();
        }
    }

    public int getChessboardWidth() {
        return chessboardWidth;
    }

    public int getChessboardHeight() {
        return chessboardHeight;
    }
    
    public CoordinatePair calculateRightHandDiagonalIntercept(CoordinatePair pointToEvaluate)
    {
        int y = pointToEvaluate.getCol();
        int x = pointToEvaluate.getRow();
        

        return pairToReturn;
    }

    public CoordinatePair calculateLeftHandDiagonalIntercept(CoordinatePair pointToEvaluate)
    {
        int y = pointToEvaluate.getCol();
        int x = pointToEvaluate.getRow();
        CoordinatePair pairToReturn = new CoordinatePair(5, 5); 


        
        
        return pairToReturn;
    }

    public boolean isChessboardValid(CoordinatePair givenCoordinatePair)
    {
        boolean validityResult = true;

        //check column
        for(int row = 0; row<this.chessboardHeight; row++)
        {
            if(this.isQueen(givenCoordinatePair.getCol(), row) && row != givenCoordinatePair.getRow() && validityResult != true)
            {
                validityResult = false;       
            }
        }
        //check row
        for(int col = 0; col<this.chessboardWidth; col++)
        {
            if(this.isQueen(col, givenCoordinatePair.getRow()) && col != givenCoordinatePair.getCol() && validityResult != true)
            {
                validityResult = false;
            }
        }
        //check right-travel diagonal
        CoordinatePair rightTravelDiagonal_StartIndex = calculateRightHandDiagonalIntercept(givenCoordinatePair);

        //check left-travel diagonal
        CoordinatePair leftTravelDiagonal_StartIndex = ;

        return validityResult;
            
    }

    public void addQueenPiece(int col, int row)
    {
        if (this.isNoQueen(col, row))
        {
            chessboard[col][row] = 1;
        }
    }

    public void removeQueenPiece(int col, int row)
    {
        if (this.isQueen(col, row))
        {
            chessboard[col][row] = 0;
        }
    }

    public boolean isNoQueen(int col, int row){
        boolean isEmpty = true;

        if(this.chessboard[col][row] == 1)
        {
            isEmpty = false;
        }

        return isEmpty;
    }

    public boolean isQueen(int col, int row){
        boolean isEmpty = false;

        if(this.chessboard[col][row] == 1)
        {
            isEmpty = true;
        }

        return isEmpty;

    

}
}