public class SnackDispenseHandler extends BasicHandler{
    private Snack snack;
    public SnackDispenseHandler(Snack givenSnack, BasicHandler next) {
        super(givenSnack, next);
        this.snack = givenSnack;
    }

    @Override
    public Snack handleRequest(String requestType) {
        Snack toReturn = null;

        if (this.snack.getQuantity() == 0)
        {
            System.out.printf("\nWe are out of the snack: %s\n", this.snack.getName());
        }else {
            if (requestType.equals(this.snack.getName()))
            {
                toReturn = this.snack;
                System.out.printf("\nHandling dispense of %s\n", this.snack.getName());
                this.snack.snackDispensed();

            }else {
                System.out.printf("\nRequest Passed from %s's handler\n", this.snack.getName());
                toReturn = super.handleRequest(requestType);
            }
        }

        return toReturn;
    }
}
