package CS4200;

//  Array[Col][Row]

public class QueensProblem
{
    public static class ChessboardPiece
    {
        private int x;
        private int y;

        public ChessboardPiece(int xCoord, int yCoord)
        {
            this.x = xCoord;
            this.y = yCoord;
        }
        public int getChessboardPiece_XCoord()
        {
            return this.x;
        }
        public int getChessboardPiece_YCoord()
        {
            return this.y;
        }
        public void setChessboardPiece_XCoord(int xCoord)
        {
            this.x = xCoord;
        }
        public void setChessboardPiece_YCoord(int yCoord)
        {
            this.y = yCoord;
        }
    }
    public static class Chessboard
    {
        private int[][]boardSpace;

        public Chessboard()
        {
            this.boardSpace = new int[8][8];
        }
        public void drawCurrentChessboardStateInConsole()
        {
            for(int colElement = 0; colElement < this.boardSpace.length; colElement++)
            {
                for(int rowElement = 0; rowElement < this.boardSpace[0].length; rowElement++)
                {
                    if(this.boardSpace[colElement][rowElement] == 0)
                    {
                        System.out.printf("\t[ ]\t");
                    }else{
                        System.out.printf("\t[x]\t");
                    }
                }
                System.out.println();
            }
        }
        private ChessboardPiece makeNewChessboardPiece(int xCoord, int yCoord)
        {
            return new ChessboardPiece(xCoord, yCoord);
        }
        private void placePieceAtCoord(int[][] inputChessboard, ChessboardPiece pieceToMove, int xCoord, int yCoord)
        {
            inputChessboard[xCoord][yCoord] = 1;
        }
        private boolean noQueensInColumn(int[][] inputChessboard, ChessboardPiece currentPiece)
        {
            boolean searchResult = true;

            for(int colElement = 0; colElement<inputChessboard.length; colElement++)
            {
                if(inputChessboard[colElement][currentPiece.getChessboardPiece_XCoord()] == 1)
                {
                    searchResult = false;
                }
            }

            return searchResult;
        }
        private ChessboardPiece findOpenSpace(int[][] inputChessboard, ChessboardPiece currentPiece)
        {
            ChessboardPiece openSpace = new ChessboardPiece(0, 0);

            for(int rowElement = 0; rowElement<inputChessboard[0].length; rowElement++)
            {
                if(noQueensInColumn(inputChessboard, currentPiece))
                {
                    openSpace.setChessboardPiece_XCoord(rowElement);
                    openSpace.setChessboardPiece_YCoord(currentPiece.getChessboardPiece_YCoord()+1);
                }   
            }

            return openSpace;
        }
        private int[][] copyBoard(int[][] boardBeingCopied)
        {
            int[][] boardToCopyTo = new int[boardBeingCopied.length][boardBeingCopied[0].length];

            for(int xElement = 0; xElement<boardBeingCopied.length; xElement++)
            {
                for(int yElement = 0; yElement<boardBeingCopied[0].length; yElement++)
                {
                    boardToCopyTo[xElement][yElement] = boardBeingCopied[xElement][yElement];
                }
            }  

            return boardToCopyTo;
        }
        private int[][] tryToPlaceQueen(ChessboardPiece testPiece)
        {
            int[][] temporaryChessboard = copyBoard(boardSpace);
            ChessboardPiece openSpace = findOpenSpace(temporaryChessboard, testPiece);

            placePieceAtCoord(temporaryChessboard, testPiece, openSpace.getChessboardPiece_XCoord(), openSpace.getChessboardPiece_YCoord());

            return temporaryChessboard;
        } 
        private boolean isChessboardSuccessful(int[][] boardToCheck)
        {
            boolean checkResult = true;

            for(int yElement = 0; yElement<boardToCheck.length; yElement++)
            {
                for(int xElement = 0; xElement<boardToCheck[0].length; xElement++)
                {
                    if(noQueensInColumn(boardToCheck, makeNewChessboardPiece(yElement, xElement)))
                    {
                        checkResult = false;
                    }
                }
            }

            return checkResult;
        }
        public void beingPlacingQueens(ChessboardPiece lastQueenPiece)
        {
            drawCurrentChessboardStateInConsole();

            ChessboardPiece newChessboardPiece = makeNewChessboardPiece(lastQueenPiece.getChessboardPiece_XCoord(), lastQueenPiece.getChessboardPiece_YCoord());
            this.boardSpace = copyBoard(tryToPlaceQueen(newChessboardPiece));

            if(isChessboardSuccessful(boardSpace))
            {
                System.out.printf("\n***[Successful Board]***");
            }else{
                beingPlacingQueens(lastQueenPiece);
            }
            
        }
    }


    public static void main(String[] args)
    {
        Chessboard board = new Chessboard();
        board.beingPlacingQueens(new ChessboardPiece(0, 0));
    }
}