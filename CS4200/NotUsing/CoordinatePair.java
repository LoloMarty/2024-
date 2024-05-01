package CS4200;

public class CoordinatePair
{
    private int col;
    private int row;

    public CoordinatePair(int givenCol, int givenRow)
    {
        this.col = givenCol;
        this.row = givenRow;
    }

    public int getCol() {
        return col;
    }

    public void setCol(int xCoord) {
        this.col = xCoord;
    }

    public int getRow() {
        return row;
    }

    public void setRow(int yCoord) {
        this.row = yCoord;
    }    

}