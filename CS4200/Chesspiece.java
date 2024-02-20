package CS4200;


public class Chesspiece
    {
        private int x;
        private int y;

        public Chesspiece(int xCoord, int yCoord)
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
