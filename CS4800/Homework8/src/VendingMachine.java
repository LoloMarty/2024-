import java.util.LinkedList;
public class VendingMachine {
    private StateOfVendingMachine vendState;
    private SnackDispenseHandler headHandler;
    private Snack selectedSnack;

    public VendingMachine()
    {
        this.vendState = new StateOfVendingMachine();
        this.headHandler = new SnackDispenseHandler(null, null);
        this.selectedSnack = null;
    }

    public void setHeadHandler(SnackDispenseHandler givenHandler)
    {
        this.headHandler = givenHandler;
    }

    public Snack getSelectedSnack()
    {
        return this.selectedSnack;
    }

    public void selectSnack(String snackName)
    {
        if(this.vendState.getState().equals("Waiting For Money"))
        {
            System.out.println("Aborting Prior Snack Order");
        }else if(this.vendState.getState().equals("Dispensing Snack"))
        {
            System.out.println("Cannot Select A New Snack At This Time.");
            return;
        }
        selectedSnack = headHandler.handleRequest(snackName);

        if (selectedSnack == null)
        {
            System.out.println("Snack Does Not Exist.");
        }else{

            this.vendState.setState_WaitForMoney();
        }
    }

    public int insertMoney(int money)
    {
        int amountToReturn = money;

        if(this.vendState.getState().equals("Idle"))
        {
            System.out.println("Please Select An Available Snack First.");
        }else if(this.vendState.getState().equals("Dispensing Snack")){
            System.out.println("Cannot Select A Snack At This Time.");
        }else{
            if (money >= selectedSnack.getPrice())
            {
                this.vendState.setState_DispensingSnack();
                System.out.printf("\nDispensing Snack: %s\n", this.selectedSnack.getName());
                amountToReturn = Math.abs(this.selectedSnack.getPrice() - money);

                this.vendState.dispenseSnack();
                System.out.println(this.selectedSnack.getName());
                System.out.printf("\nChange Due: %d", amountToReturn);
            }else{
                System.out.printf("\nInsufficient Funds for Snack: %s", this.selectedSnack.getName());
            }

            this.vendState.setState_Idle();
        }

        return amountToReturn;
    }

}

