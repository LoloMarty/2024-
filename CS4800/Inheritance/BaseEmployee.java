package CS4800.Inheritance;

public class BaseEmployee extends Employee {
    private int baseSalary;

    public BaseEmployee(String givenFirstName, String givenLastName, int givenSocialSecurityNumber,
            int givenBaseSalary) {

        super(givenFirstName, givenLastName, givenSocialSecurityNumber);
        this.baseSalary = givenBaseSalary;
    }

    public void setBaseSalary(int newBaseSalary) {
        this.baseSalary = newBaseSalary;
    }

    public int getBaseSalary() {
        return this.baseSalary;
    }
}
