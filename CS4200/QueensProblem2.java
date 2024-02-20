package CS4200;

public class QueensProblem2 {

    public static void main(String[] args)
    {
        Chessboard invalidChessboard = new Chessboard(8, 8);

        invalidChessboard.addQueenPiece(0, 1);
        invalidChessboard.addQueenPiece(0, 3);

        System.out.printf("\nValidity: %b", invalidChessboard.isChessboardValid());

        invalidChessboard.removeQueenPiece(0, 3);

        System.out.printf("\nValidity: %b", invalidChessboard.isChessboardValid());
    }

}
