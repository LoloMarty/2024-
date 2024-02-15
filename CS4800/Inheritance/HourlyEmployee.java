package CS4800.Inheritance;

public class HourlyEmployee extends Employee {
    private int wage;
    private int hoursWorked;

    public HourlyEmployee(String givenFirstName, String givenLastName, int givenSocialSecurityNumber, int givenWage,
            int givenHoursWorked) {

        super(givenFirstName, givenLastName, givenSocialSecurityNumber);
        this.wage = givenWage;
        this.hoursWorked = givenHoursWorked;
    }

    public void setWage(int newWage) {
        this.wage = newWage;
    }

    public int getEmployeeWage() {
        return this.wage;
    }

    public void setHoursWorked(int newAmountOfHoursWorked) {
        this.hoursWorked = newAmountOfHoursWorked;
    }

    public int getHoursWorked() {
        return this.hoursWorked;
    }
}
