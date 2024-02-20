package CS4200;

//  Array[Col][Row]

public class Chessboard {
    private int[][] chessboard;
    private int chessboardWidth;
    private int chessboardHeight;

    public int getChessboardWidth() {
        return chessboardWidth;
    }

    public int getChessboardHeight() {
        return chessboardHeight;
    }
    
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

    public boolean isChessboardValid()
    {
        boolean queenFoundInColumn = false;
        boolean validityResult = true;

        // for every col
        for (int col = 0; col < chessboard.length; col++)
        {
            // for every row
            for(int row = 0; row < chessboard[0].length; row++)
            {
                if(this.isQueen(col, row))
                {
                    if(queenFoundInColumn == false)
                    {
                        queenFoundInColumn = true;
                    }else{
                        validityResult = false;
                    }
                }
            }

            queenFoundInColumn = false;
        }

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