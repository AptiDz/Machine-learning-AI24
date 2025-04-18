# Impotering av bibliotek
# - fuzzywussy: sträng-matchning för att kunna hitta korrekt titel på filmen
from fuzzywuzzy import process

# Skapar funktionen som ska returnera titel på filmen via dataframe och ID på filmen
def movie_name(df_movies, id):
    return df_movies.query("movieId == @id")["title"].item()

# Funktionen som ska be användare att skriva in filmtitel eller tag och få ut rekommenderade titlar via dataframe rekommenderbar filmer, dataframe tags, dataframe movies, skapade KNN model och gles matris.
# - Genom fuzzywuzzy kan man ta fram de 6 närmaste grannarna via kneighbors, 6 för att inmattade är inkluderat
# - Skriva ut de 5 närmaste rekommendationer
def movie_recommendations(df_recommendations, df_tags, df_movies, KNN_model, sparse_matrix):
    # Ber användaren att skriva en filmtitel
    user_input = input("Skriv in film titel eller tag: ")

    # process.extractOne ska returnera bästa matchningen mot dataframen rekommenderbar filmer
    # Sätter score_cutoff till 90 så att matchningens poäng är minst 90 på 0-100 skala.
    match = process.extractOne(user_input, df_recommendations["title"], score_cutoff=90)
    
    # Om titel matchar då används matchade filmens index för KNN-sökning
    if match is not None:
            
        # När det inte finns matchning exempelvis score är under 90 ska det bli None fast om det lyckas så ska det fram index i position 2
        # process.extractOne returnera tuple med 3 elemeter matchande titel, matchningspoäng och index.
        index = match[2]
        
        # Hämtar index på de 6 närmaste grannarna. Kneighbors returnera distans och index och det är indexet som är relevant nu.
        _ , indices = KNN_model.kneighbors(sparse_matrix[index], n_neighbors = 6)

        # För att skriva ut inputfilmen. [0][0] för att gå in i första elementen vilket är lista med närmsta grannarnas index och dess första index ska representera filmen man matchade med fuzzywuzzy.
        print(f"Top 5 rekommendationer för: {df_recommendations['title'].iloc[indices[0][0]]}")
        
        # Loopar igenom de övriga 5 indexen och skriver ut deras titlar som rekommendationer
        for neighbor_index in indices[0][1:]:
            print(df_recommendations["title"].iloc[neighbor_index])
    
    # Om ingen titel hittas då försöker man i tags_df istället
    else:
        # Skapar en lista med unika taggar via dataframen tags
        unique_tags = df_tags["tag"].unique()
        tag_match = process.extractOne(user_input, unique_tags, score_cutoff=90)
        
        if tag_match is not None:
            matching_tag = tag_match[0]
            
            # Hämta alla movieIDs kopplade till den matchade tag.
            tag_movies = df_tags[df_tags["tag"] == matching_tag]["movieId"]
            # Räkna antalet gånger tags finns per movieId och välj det med högst antal 
            movie_counts = tag_movies.value_counts()
            most_common_movie_id = movie_counts.idxmax()
            
            # Hitta samma index i dataframen rekommenderbar som motsvarar detta movieID
            try:
                new_index = df_recommendations.index[df_recommendations["movieId"] == most_common_movie_id][0]
            except (TypeError, IndexError):
                print(f"Beklagar! Kan inte hitta titel eller tag för '{user_input}' i databasen.")
                return
            
            print(f"Hittade via tag: '{matching_tag}'")
            
            # Hämtar filmens titel via tidigare movies_name funktion
            movie_title = movie_name(df_movies, most_common_movie_id)
            print(f"Valde filmen baserat på taggen: {movie_title}")
            
            # Kör KNN på den hittade filmen via taggen
            _, indices = KNN_model.kneighbors(sparse_matrix[new_index], n_neighbors=6)
            print(f"Top 5 rekommendationer för: {df_recommendations['title'].iloc[new_index]}")
            for neighbor_index in indices[0][1:]:
                print(df_recommendations["title"].iloc[neighbor_index])
        else:
            print(f"Tyvärr sökning med '{user_input}' funkade inte, Prova med annan inmattning.")
