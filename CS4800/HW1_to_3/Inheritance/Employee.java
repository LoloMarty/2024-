package CS4800.Inheritance;

public class Employee {
    private String firstName;
    private String lastName;
    private int socialSecurityNumber;

    public Employee(String givenFirstName, String givenLastName, int givenSocialSecurityNumber) {
        this.firstName = givenFirstName;
        this.lastName = givenLastName;
        this.socialSecurityNumber = givenSocialSecurityNumber;
    }

    public void setEmployeeFirstName(String newName) {
        this.firstName = newName;
    }

    public String getEmployeeFirstName() {
        return this.firstName;
    }

    public void setEmployeeLastName(String newName) {
        this.lastName = newName;
    }

    public String getEmployeeLastName() {
        return this.lastName;
    }

    public void setEmployeeSocialSecurityNumber(int newSSN) {
        this.socialSecurityNumber = newSSN;
    }

    public int getEmployeeSocialSecurityNumber() {
        return this.socialSecurityNumber;
    }

}