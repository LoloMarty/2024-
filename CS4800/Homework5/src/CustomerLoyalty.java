public class CustomerLoyalty extends FoodBase{
    final double oneYearDiscountRate = 0.15;
    final double twoYearDiscountRate = 0.5;
    final double threeYearDiscountRate = 0.9;
    public CustomerLoyalty(IFood wrapped)
    {
        super(wrapped);
    }


    public int calculateCost(int CustomerLoyaltyYear) {
        int totalCost = super.calculateCost();
        double appliedDiscountRate = 0;

        if(CustomerLoyaltyYear == 3)
        {
            appliedDiscountRate = 1-this.threeYearDiscountRate;
        }else if (CustomerLoyaltyYear == 2)
        {
            appliedDiscountRate = 1-this.twoYearDiscountRate;
        }else if(CustomerLoyaltyYear == 1)
        {
            appliedDiscountRate = 1-this.oneYearDiscountRate;
        }else{
            appliedDiscountRate = 1;
        }

        System.out.printf("\n\nTotal Cost: %d\nTotal Cost W/ DISCOUNT: %d", (int)totalCost, (int)(totalCost * appliedDiscountRate));

        return (int)(totalCost * appliedDiscountRate);
    }
}
