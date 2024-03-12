package AbstractFactory;

public class Customer {
    private String name;
    private DietPlan plan;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public DietPlan getPlan() {
        return plan;
    }

    public void setPlan(DietPlan plan) {
        this.plan = plan;
    }
}
