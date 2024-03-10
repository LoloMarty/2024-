package CS4800.Inheritance;

public class DriverProgram {

    public DriverProgram(Employee firstEmployee) {

    }

    private String convertIntToFormattedSSN(int givenSocialSecurityNumber) {
        String formattedSSN = "";

        formattedSSN += (Integer.toString(givenSocialSecurityNumber)).substring(0, 3);
        formattedSSN += "-";
        formattedSSN += (Integer.toString(givenSocialSecurityNumber)).substring(3, 5);
        formattedSSN += "-";
        formattedSSN += (Integer.toString(givenSocialSecurityNumber)).substring(5, 9);

        return formattedSSN;
    }

    private void printEmployeeInformationTableHeader_Inline() {
        System.out.printf(
                "\n|First Name\t|Last Name\t|Social Sec #\t| Weekly Salary\t|Wage\t|Hours Worked\t|Com Rate\t|Gross Salary\t|Base Salary\t|");
    }

    private void printSalariedEmployeeInformation_Inline(SalariedEmployee employeeInfoToPrint) {
        System.out.printf("\n|%-15s|%-15s|%s\t|$%,d\t\t|  \t|  \t\t|  \t\t|  \t\t|  \t\t|",
                employeeInfoToPrint.getEmployeeFirstName(),
                employeeInfoToPrint.getEmployeeLastName(),
                this.convertIntToFormattedSSN(employeeInfoToPrint.getEmployeeSocialSecurityNumber()),
                employeeInfoToPrint.getWeeklySalary());
    }

    private void printHourlyEmployeeInformation_Inline(HourlyEmployee employeeInfoToPrint) {
        System.out.printf("\n|%-15s|%-15s|%s\t|  \t\t|$%,d\t|%s\t\t|  \t\t|  \t\t|  \t\t|",
                employeeInfoToPrint.getEmployeeFirstName(),
                employeeInfoToPrint.getEmployeeLastName(),
                this.convertIntToFormattedSSN(employeeInfoToPrint.getEmployeeSocialSecurityNumber()),
                employeeInfoToPrint.getEmployeeWage(),
                Integer.toString(employeeInfoToPrint.getHoursWorked()));
    }

    private void printCommisionEmployeeInformation_Inline(CommisionEmployee employeeInfoToPrint) {
        System.out.printf("\n|%-15s|%-15s|%s\t|  \t\t|  \t|  \t\t|%s%%\t\t|$%,d\t|  \t\t|",
                employeeInfoToPrint.getEmployeeFirstName(),
                employeeInfoToPrint.getEmployeeLastName(),
                this.convertIntToFormattedSSN(employeeInfoToPrint.getEmployeeSocialSecurityNumber()),
                Integer.toString(employeeInfoToPrint.getCommisionRate()),
                employeeInfoToPrint.getGrossSales());
    }

    private void printBaseEmployeeInformation_Inline(BaseEmployee employeeInfoToPrint) {
        System.out.printf("\n|%-15s|%-15s|%s\t|  \t\t|  \t|  \t\t|  \t\t|  \t\t|$%,d\t|",
                employeeInfoToPrint.getEmployeeFirstName(),
                employeeInfoToPrint.getEmployeeLastName(),
                this.convertIntToFormattedSSN(employeeInfoToPrint.getEmployeeSocialSecurityNumber()),
                employeeInfoToPrint.getBaseSalary());
    }

    public static void main(String[] args) {

        SalariedEmployee joeJones = new SalariedEmployee("Joe", "Jones", 111111111, 2500);
        HourlyEmployee stephSmith = new HourlyEmployee("Stephanie", "Smith", 222222222, 25, 32);
        HourlyEmployee maryQuinn = new HourlyEmployee("Mary", "Quinn", 333333333, 19, 47);
        CommisionEmployee nicoleDior = new CommisionEmployee("Nicole", "Dior", 444444444, 15, 50000);
        SalariedEmployee renwaChanel = new SalariedEmployee("Renwa", "Chanel", 555555555, 1700);
        BaseEmployee mikeDavenPort = new BaseEmployee("Mike", "Davenport", 666666666, 95000);
        CommisionEmployee mahnazVaziri = new CommisionEmployee("Mahnaz", "Vaziri", 777777777, 22, 40000);

        DriverProgram driver = new DriverProgram(joeJones);

        driver.printEmployeeInformationTableHeader_Inline();
        driver.printSalariedEmployeeInformation_Inline(joeJones);
        driver.printHourlyEmployeeInformation_Inline(stephSmith);
        driver.printHourlyEmployeeInformation_Inline(maryQuinn);
        driver.printCommisionEmployeeInformation_Inline(nicoleDior);
        driver.printSalariedEmployeeInformation_Inline(renwaChanel);
        driver.printBaseEmployeeInformation_Inline(mikeDavenPort);
        driver.printCommisionEmployeeInformation_Inline(mahnazVaziri);
    }
}