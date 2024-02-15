package CS4800.Inheritance;

public class CommisionEmployee extends Employee {
    private int commisionRate;
    private int grossSales;

    public CommisionEmployee(String givenFirstName, String givenLastName, int givenSocialSecurityNumber,
            int givenCommisionRate, int givenGrossSales) {

        super(givenFirstName, givenLastName, givenSocialSecurityNumber);
        this.commisionRate = givenCommisionRate;
        this.grossSales = givenGrossSales;
    }

    public void setCommisionRate(int newCommisionRate) {
        this.commisionRate = newCommisionRate;
    }

    public int getCommisionRate() {
        return this.commisionRate;
    }

    public void setGrossSales(int newGrossSaleAmount) {
        this.grossSales = newGrossSaleAmount;
    }

    public int getGrossSales() {
        return this.grossSales;
    }
}
