public class StateOfVendingMachine {
    private String state;

    public StateOfVendingMachine()
    {
        this.state = "Idle";
    }

    public String getState() {
        return state;
    }

    public void setState_Idle()
    {
        if(!this.state.equals("Idle"))
        {
            System.out.printf("\nChanging To State: Idle", this.state);
        }
            this.state = "Idle";
    }

    public void setState_WaitForMoney()
    {
        if(!this.state.equals("Waiting For Money"))
        {
            System.out.println("Changing to State: Waiting For Money");
        }
            this.state = "Waiting For Money";
    }

    public void setState_DispensingSnack()
    {
        if(!this.state.equals("Dispensing Snack"))
        {
            System.out.println("Changing to State: Dispensing Snack");
        }
            this.state = "Dispensing Snack";
    }

    public void dispenseSnack()
    {
        System.out.println("Please retrieve your snack from the receptacle:");
    }
}
