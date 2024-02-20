package CS4200;

//  Array[Col][Row]

public class GameTreeDriver {
    
    private Chessboard chessboard;

    public GameTreeDriver(int givenChessboardWidth, int givenChessboardHeight)
    {
   
        this.chessboard = new Chessboard(givenChessboardWidth, givenChessboardHeight);
    }

    private CoordinatePair findAvailableSpotInRow(int row, int startingCol)
    {
        CoordinatePair coordToReturn = null;
        Boolean coordinateNotFound = true;

        for(int col = startingCol; col<this.chessboard.getChessboardWidth(); col++)
        {
            if(this.chessboard.isNoQueen(col, row) && coordinateNotFound)
            {
                coordToReturn = new CoordinatePair(col, row);
                coordinateNotFound = false;
            }
        }

        return coordToReturn;
    }

    private boolean tryToPlacePiece(CoordinatePair coordinateToPlace)
    {
        this.chessboard.addQueenPiece(coordinateToPlace.getCol(), coordinateToPlace.getRow());

        return this.chessboard.isChessboardValid();
    }

    public boolean bruteForceMethod(CoordinatePair coordinateToStartAt)
    {
        boolean treeProgressionSuccess = false;

        CoordinatePair openSpot = this.findAvailableSpotInRow(coordinateToStartAt.getRow(), coordinateToStartAt.getCol());
        boolean placementSuccess = tryToPlacePiece(openSpot);

        this.chessboard.printChessboardOnConsole();

        if(placementSuccess == true)
        {
            for(int col = 0; col < this.chessboard.getChessboardWidth(); col++)
            {
                treeProgressionSuccess = true;
                bruteForceMethod(new CoordinatePair(col, coordinateToStartAt.getRow()+1));
            }
        }else{
            treeProgressionSuccess = false;
            this.chessboard.removeQueenPiece(openSpot.getCol(), openSpot.getRow());
            
        }

        return treeProgressionSuccess;
        
    }

    public static void main(String[] args)
    {
        
        CoordinatePair startingCoordinates = new CoordinatePair(0, 0);

        GameTreeDriver driver = new GameTreeDriver(8, 8);
        driver.bruteForceMethod(startingCoordinates);
        

        /* 
        Chessboard invalidChessboard = new Chessboard(8, 8);
        invalidChessboard.addQueenPiece(0, 0);
        invalidChessboard.addQueenPiece(1, 1);
        invalidChessboard.printChessboardOnConsole();
        System.err.println(invalidChessboard.isChessboardValid());
        */
    }

}
