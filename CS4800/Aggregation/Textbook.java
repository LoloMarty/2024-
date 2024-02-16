package CS4800.Aggregation;

public class Textbook {
    private String title;
    private String author;
    private String publisher;

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public String getPublisher() {
        return publisher;
    }

    public void setPublisher(String publisher) {
        this.publisher = publisher;
    }

    public Textbook(String givenTitle, String givenAuthor, String givenPublisher) {
        this.setTitle(givenTitle);
        this.setAuthor(givenAuthor);
        this.setPublisher(givenPublisher);
    }

}
