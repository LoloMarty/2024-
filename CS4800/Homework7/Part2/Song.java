public class Song {
    private final String title;
    private final String artist;
    private final String album;
    private final int id;
    private final int duration;

    public Song(String givenTitle, String givenArtist, String givenAlbum, int givenID, int givenDuration) {
        this.title = givenTitle;
        this.artist = givenArtist;
        this.album = givenAlbum;
        this.id = givenID;
        this.duration = givenDuration;
    }

    public String getTitle() {
        return title;
    }

    public String getArtist() {
        return artist;
    }

    public String getAlbum() {
        return album;
    }

    public int getId() {
        return id;
    }

    public int getDuration() {
        return duration;
    }

}
