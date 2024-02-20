package CS4200;

//  Array[Col][Row]

public class Chessboard {
    private int[][] chessboard;

    public Chessboard(int dimX, int dimY)
    {
        this.chessboard = new int[dimX][dimY];
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
        }

        return validityResult;
            
    }

    public void addQueenPiece(int xCoord, int yCoord)
    {
        if (this.isNoQueen(xCoord, yCoord))
        {
            chessboard[xCoord][yCoord] = 1;
        }
    }

    public void removeQueenPiece(int xCoord, int yCoord)
    {
        if (this.isQueen(xCoord, yCoord))
        {
            chessboard[xCoord][yCoord] = 0;
        }
    }

    public boolean isNoQueen(int xCoord, int yCoord){
        boolean isEmpty = true;

        if(this.chessboard[xCoord][yCoord] == 1)
        {
            isEmpty = false;
        }

        return isEmpty;
    }

    public boolean isQueen(int xCoord, int yCoord){
        boolean isEmpty = false;

        if(this.chessboard[xCoord][yCoord] == 1)
        {
            isEmpty = true;
        }

        return isEmpty;

    

}
}