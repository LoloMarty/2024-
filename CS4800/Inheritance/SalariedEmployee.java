package CS4800.Inheritance;

public class SalariedEmployee extends Employee {
    private int weeklySalary;

    public SalariedEmployee(String givenFirstName, String givenLastName, int givenSocialSecurityNumber,
            int givenWeeklySalary) {

        super(givenFirstName, givenLastName, givenSocialSecurityNumber);
        this.weeklySalary = givenWeeklySalary;
    }

    public void setWeeklySalary(int newWeeklySalary) {
        this.weeklySalary = newWeeklySalary;
    }

    public int getWeeklySalary() {
        return this.weeklySalary;
    }
}