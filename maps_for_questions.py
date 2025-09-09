answer_map = {}
answer_map["Q2"] = {
    1: "Da",
    2: "Ne",
    3: "Smo v postopku izdelave spletne strani",
}
answer_map["Q3"] = {
    1: "V zadnjih 3 mesecih",
    2: "V zadnjih 6 mesecih",
    3: "V zadnjem letu",
    4: "V zadnjih 2 letih",
    5: "Več kot 2 leti",
    6: "Spletna stran ni bila nikoli posodobljena",
}
answer_map["Q4"] = {
    1: "Nikoli",
    2: "Nekajkrat letno",
    3: "Enkrat mesečno",
    4: "Enkrat na teden",
    5: "Večkrat na teden",
}
answer_map["Q5"] = {
    1: "Da",
    2: "Ne",
}
answer_map["Q6"] = {
    1: "Microsoft Teams",
    2: "Zoom",
    3: "Drugo",
}
answer_map["Q7"] = {
    1: "Da",
    2: "Ne",
}
answer_map["Q8"] = {
    1: "Da",
    2: "Ne",
}
answer_map["Q9"] = {
    1: "Starešina",
    2: "Pomočnik/namestnik starešine",
    3: "Gospodar",
    4: "Pomočnik/namestnik gospodarja",
    5: "Tajnik",
    6: "Informatik",
    7: "Ostali",
}
answer_map["Q10"] = {
    1: "Zelo pozitivno",
    2: "Pozitivno",
    3: "Negativno",
    4: "Zelo negativno",
}
answer_map["Q11"] = {
    1: "Da",
    2: "Ne",
}
answer_map["Q12"] = {
    1: "Zelo koristno",
    2: "Koristno",
    3: "Nekoristno",
    4: "Zelo nekoristno",
}
answer_map["Q13"] = {
    1: "Zelo koristno",
    2: "Koristno",
    3: "Nekoristno",
    4: "Zelo nekoristno",
}
answer_map["Q14"] = {
    1: "Zelo koristno",
    2: "Koristno",
    3: "Nekoristno",
    4: "Zelo nekoristno",
}
answer_map["Q16"] = {
    1: "Rezervacija preže",
    2: "Označitev zasedenosti in sprostitev zasedenosti preže",
    3: "Prikaz trenutne lokacije lovca v lovišču",
    4: "Prikaz QR kode od lovske izkaznice",
    5: "Vnos osnovnih podatkov o uplenu",
    6: "Prikaz zasedenosti prež na zemljevidu",
    7: "Vodenje dnevnika lovskega pripravnika",
    8: "Dodatna zaželjena funkcionalnost",
}
question_map = {
    "Q2": "Ali ima vaša lovska družina spletno stran?",
    "Q3": "Kdaj ste nazadnje posodobili spletno stran?",
    "Q4": "Kako pogosto izvajate sestanke na daljavo preko spleta (Zoom, MS Teams)?",
    "Q5": "Ali bi izvajali sestanke na daljavo v primeru, če bi Lovska zveza Slovenije za vašo lovsko družino poskrbela za to možnost?",
    "Q6": "Kaj uporabljate za sestanke na daljavo?",
    "Q7": "Ali bi bilo po vašem mnenju dobro, da bi lovska družina imela (spletno) aplikacijo za podporo delovanja lovske družine?",
    "Q8": "Ali bi bili pripravljeni preiti na uporabo nove spletne strani, ki bi direktno uporabljala nekatere podatke iz aplikacije za podporo delovanja lovske družine?",
    "Q9": "Kdo v lovski družini uporablja aplikacijo Lisjak?",
    "Q10": "Kakšno je splošno mnenje v lovski družini o aplikaciji Lisjak?",
    "Q11": "Ali bi bilo po vašem mnenju dobro, da bi lovci imeli možnost za izobraževanje na daljavo?",
    "Q12": "Ali bi bilo po vašem mnenju koristno, da bi aplikacija Lisjak predčasno poslala lovski družini opozorilo o izteku veljavnosti orožnega lista, izteku veljavnosti imenovanja lovskega čuvaja in izteku veljavnosti pregledniškega tečaja za pregled trupa divjadi za posamezne člane lovske družine?",
    "Q13": "Ali bi bilo po vašem mnenju koristno, da bi imeli lovski čuvaji aplikacijo za izdelavo poročil?",
    "Q14": "Ali bi bilo po vašem mnenju koristno, da bi aplikacija Lisjak opozorila na nepravilno prijavo za odlikovanja?",
    "Q15": "Katere podatke v aplikaciji Lisjak pogrešate in bi jih bilo po vašem mnenju Lisjak moral imeti?",
    "Q16": "Katere ključne funkcionalnosti bi po vašem mnenju morala imeti mobilna aplikacija za lovce?",
}

# Q2, Q3, Q4, Q5, Q7, Q8, Q10, Q11, Q12, Q13, Q14 (11/15) ############################################################################################################
indexes_for_single_choice_questions = [0, 1, 2, 3, 8, 9, 18, 19, 20, 21, 22]
