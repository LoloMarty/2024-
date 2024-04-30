import org.junit.Test;

import static org.junit.Assert.*;

public class SongTest {

    @Test
    public void getTitle() {
        String givenTitle = "Bohemian Rhapsody";
        String givenArtist = "Queen";
        String givenAlbum = "A Night at the Opera";
        int givenID = 1;
        int givenDuration = 355;

        Song song = new Song(givenTitle, givenArtist, givenAlbum, givenID, givenDuration);

        assertEquals(givenTitle, song.getTitle());
    }

    @Test
    public void getArtist() {
        String expectedArtist = "Ed Sheeran";
        Song song = new Song("Shape of You", expectedArtist, "÷", 1, 234);

        String actualArtist = song.getArtist();

        assertEquals(expectedArtist, actualArtist);
    }

    @Test
    public void getAlbum() {
        Song song = new Song("Title", "Artist", "Album", 1, 180);
        assertEquals("Album", song.getAlbum());
    }

    @Test
    public void getId() {
        String givenTitle = "Song Title";
        String givenArtist = "Artist Name";
        String givenAlbum = "Album Title";
        int givenID = 123;
        int givenDuration = 240;

        Song song = new Song(givenTitle, givenArtist, givenAlbum, givenID, givenDuration);

        assertEquals(givenID, song.getId());
    }

    @Test
    public void getDuration() {
        int expectedDuration = 240;

        Song song = new Song("Title", "Artist", "Album", 1, expectedDuration);
        assertEquals(expectedDuration, song.getDuration());
    }
}