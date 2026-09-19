# === f1_env.py (stabilna + robusna na raznolike scenarije - kiša/SC/VSC) ===
import gym
from gym import spaces
import numpy as np
from utils import selected_features, scale_observation

# Curriculum sad ne filtrira retke po Weatheru (to više nema smisla otkad je
# Weather stvaran i uvijek suh za dostupne utrke) - umjesto toga kontrolira
# KOLIKO ČESTO agent tijekom treninga nailazi na sintetički ubačene, ali
# vremenski DOSLJEDNE (blokovske, ne po-krugu) scenarije kiše/vlage i
# Safety Car/VSC preko stvarnih suhih podataka. Cilj: agent koji zna voziti
# bilo koju utrku, ne samo one tri suhe koje su nam stvarno dostupne.
DIFFICULTY_AUGMENT_PARAMS = {
    "easy":   {"p_weather_event": 0.015, "p_caution_event": 0.010},
    "medium": {"p_weather_event": 0.040, "p_caution_event": 0.025},
    "full":   {"p_weather_event": 0.070, "p_caution_event": 0.040},
}

class FastF1MultiActionEnv(gym.Env):
    def __init__(self, obs_array, true_actions=None, difficulty_level="easy"):
        super().__init__()
        self.obs_array = obs_array
        self.true_actions = true_actions
        self.difficulty_level = difficulty_level
        self.index = 0
        self.previous_actions = []

        self.action_space = spaces.MultiDiscrete([2, 5, 3])  # pit, compound, style
        # Nakon uvođenja scale_observation(), sve značajke su podijeljene sa
        # svojom skalom pa je stvarni raspon opažanja otprilike [0, ~1.5]
        # (uz mali sigurnosni razmak za rubne/augmentirane vrijednosti).
        self.observation_space = spaces.Box(
            low=0, high=2.0, shape=(obs_array.shape[1],), dtype=np.float32
        )

        self._i_weather = selected_features.index("Weather")
        self._i_trackstatus = selected_features.index("TrackStatus")
        self._i_lapnum = selected_features.index("LapNumber")

        # Generator slučajnih brojeva korišten za SVU stohastičnost u ovoj
        # klasi (miješanje redoslijeda utrka, sintetička augmentacija,
        # nasumični DNF) - vidi napomenu u reset() o tome kako seed radi.
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        # NAPOMENA (popravak reproducibilnosti): reset() je ranije primao
        # `seed` parametar ali ga NIKAD nije koristio - sva nasumičnost je
        # išla preko globalnog np.random, pa dva poziva `env.reset(seed=42)`
        # NISU davala isti rezultat (globalno stanje generatora ovisi o
        # svemu što se ranije dogodilo u procesu). To je posebno utjecalo na
        # "Vizualizacija 2/3" u train.py, gdje se svaki put koristi isti
        # seed=42/123 s namjerom usporedive evaluacije kroz epizode - bez
        # ovog popravka, svaka evaluacija je zapravo bila NA DRUGAČIJEM
        # (nasumično promiješanom/augmentiranom) scenariju, dodajući šum u
        # graf reward-a koji nije imao veze sa stvarnim napretkom treninga.
        # Sad: eksplicitan seed -> potpuno determinističan reset (isti
        # redoslijed utrka, ista sintetička augmentacija, isti DNF ishodi).
        # Bez seeda (kako to interno radi SB3 tijekom treninga) -> generator
        # nastavlja iz svog trenutnog stanja, ostajemo nasumični kao i prije.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.index = 0
        self.previous_actions = []

        shuffled, orig_indices = self._shuffle_race_order(self.obs_array)
        augmented, synth_weather, synth_caution = self._augment_scenarios(shuffled)
        self.filtered_obs_array = augmented
        # Mapiranje: pozicija u filtered_obs_array -> pozicija u self.true_actions
        # (nužno jer smo utrke poredali nasumično; bez ovoga bi se true_action
        # nagrada uspoređivala s pogrešnim krugom).
        self.orig_indices = orig_indices
        self.is_synthetic_weather = synth_weather
        self.is_synthetic_caution = synth_caution

        # Granice utrka UNUTAR (promiješanog) niza za ovu epizodu. Bitno: sam
        # niz je zapravo N spojenih stvarnih utrka, ne jedna duga utrka - sav
        # bookkeeping koji se odnosi na "cijelu utrku" (povijest guma/vremena,
        # monotony-provjere, terminalne kazne) mora se resetirati na SVAKOJ
        # granici, ne samo na kraju cijelog niza.
        self.race_segments = self._split_into_races(self.filtered_obs_array)
        self.race_starts = set(s for s, _ in self.race_segments)
        self.race_end_for = np.empty(len(self.filtered_obs_array), dtype=np.int64)
        for s, e in self.race_segments:
            self.race_end_for[s:e] = e - 1

        # Prati je li utrka do sada bila suha i koji su suhi compoundi korišteni,
        # potrebno za pravilo o obaveznih 2 tipa suhih guma (isto pravilo koje
        # safety_layer.py primjenjuje kao post-processing - ovdje ga učimo i
        # kroz reward tako da se RAW politika sama približi SAFE ponašanju).
        # Sve ovo je PO UTRCI, resetira se i na granicama utrka u step().
        self.wet_or_damp_seen = False
        self.dry_compounds_used = set()
        self.lap_in_race = 0
        return scale_observation(self.filtered_obs_array[self.index]), {}

    def _split_into_races(self, array):
        """Vrati listu (start, end) raspona indeksa za svaku zasebnu utrku/
        stint u nizu, detektirano preko pada vrijednosti LapNumber (pad =
        nova utrka/vozač). Isti princip kao segmentacija u predict_strategies.py."""
        lap_numbers = array[:, self._i_lapnum]
        starts = [0]
        for idx in range(1, len(lap_numbers)):
            if lap_numbers[idx] < lap_numbers[idx - 1]:
                starts.append(idx)
        starts.append(len(array))
        return [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]

    def _shuffle_race_order(self, array):
        """Promiješa REDOSLIJED cijelih utrka (ne pojedinačnih krugova unutar
        njih!) - ovo čuva stvarni slijed krugova unutar svake utrke, što je
        nužno da sintetički blokovi ispod (i praćenje TyreLife/Stint progresije)
        imaju ikakvog smisla, dok svejedno unosi raznolikost redoslijeda
        treniranja iz epizode u epizodu. Stara verzija (shuffle_weather) je
        miješala POJEDINAČNE retke unutar četvrtina niza, čime je uništavala
        stvarni slijed krugova - to je bila greška, ne samo za sintetičke
        scenarije nego i za temeljno praćenje trošenja guma kroz utrku.

        Vraća i mapiranje (orig_indices) na izvorne indekse u obs_array, jer
        self.true_actions je indeksiran prema IZVORNOM (neizmiješanom) redoslijedu."""
        races = self._split_into_races(array)
        order = list(range(len(races)))
        self._rng.shuffle(order)
        shuffled = np.concatenate([array[races[k][0]:races[k][1]] for k in order], axis=0)
        orig_indices = np.concatenate([np.arange(races[k][0], races[k][1]) for k in order])
        return shuffled, orig_indices

    def increase_difficulty(self):
        if self.difficulty_level == "easy":
            self.difficulty_level = "medium"
        elif self.difficulty_level == "medium":
            self.difficulty_level = "full"

    def _augment_scenarios(self, array):
        """
        Vraća KOPIJU niza s ubačenim sintetičkim, ali VREMENSKI DOSLJEDNIM
        (blokovskim, ne treperećim po-krugu kao stara np.random.choice verzija)
        scenarijima kiše/vlage i Safety Car/VSC, po svakoj utrci zasebno (blok
        nikad ne prelazi granicu između dvije stvarne utrke). Također vraća
        dvije boolean maske koje označavaju koji su krugovi sintetički
        izmijenjeni - koristimo ih u step() da ISKLJUČIMO "true_action"
        (usporedba sa stvarnim vozačem) nagradu na tim krugovima, jer stvarni
        vozač nikad nije vozio pod tim izmišljenim uvjetima pa njegova stvarna
        (suha) odluka tu više nije ispravan cilj učenja - umjesto toga se
        oslanjamo isključivo na pravila (R1/R2/R6/R9-stil) iznad.
        """
        params = DIFFICULTY_AUGMENT_PARAMS.get(self.difficulty_level, DIFFICULTY_AUGMENT_PARAMS["full"])
        arr = array.copy()
        n = arr.shape[0]
        synth_weather = np.zeros(n, dtype=bool)
        synth_caution = np.zeros(n, dtype=bool)

        for start, end in self._split_into_races(arr):
            # --- perzistentni blokovi kiše/vlage ---
            idx = start
            while idx < end:
                if self._rng.random() < params["p_weather_event"]:
                    block_len = min(int(self._rng.integers(5, 15)), end - idx)
                    is_wet = self._rng.random() < 0.35
                    arr[idx:idx + block_len, self._i_weather] = 3 if is_wet else 2
                    synth_weather[idx:idx + block_len] = True
                    idx += block_len
                else:
                    idx += 1

            # --- povremeni Safety Car / VSC / žuti (kratki blokovi) ---
            idx = start
            while idx < end:
                if self._rng.random() < params["p_caution_event"]:
                    block_len = min(int(self._rng.integers(2, 6)), end - idx)
                    code = self._rng.choice([2, 4, 6])  # žuti / Safety Car / VSC
                    arr[idx:idx + block_len, self._i_trackstatus] = code
                    synth_caution[idx:idx + block_len] = True
                    idx += block_len
                else:
                    idx += 1

        return arr, synth_weather, synth_caution

    def step(self, action):
        # Nova utrka počinje UNUTAR iste epizode (filtered_obs_array je N
        # spojenih stvarnih utrka) - resetiraj sav bookkeeping vezan uz "cijelu
        # utrku" da se ne miješa s prethodnom (inače bi npr. monotony-provjera
        # usporedila zadnji krug utrke A sa stilom vožnje iz utrke B).
        if self.index in self.race_starts and self.index != 0:
            self.wet_or_damp_seen = False
            self.dry_compounds_used = set()
            self.previous_actions = []
            self.lap_in_race = 0

        pit, comp, style = action
        self.previous_actions.append(action)

        lap = self.filtered_obs_array[self.index]
        i = lambda name: selected_features.index(name)
        lap_time = lap[i("LapTime")]
        tyre_life = lap[i("TyreLife")]
        front_gap = lap[i("FrontGap")]
        weather = lap[i("Weather")]
        track_status = lap[i("TrackStatus")] if "TrackStatus" in selected_features else 1
        is_synth_weather_lap = bool(self.is_synthetic_weather[self.index])
        is_synth_caution_lap = bool(self.is_synthetic_caution[self.index])

        # Snapshot PRIJE ažuriranja - treba nam za bonus "prvi put ovaj compound"
        self._dry_compounds_before_this_lap = set(self.dry_compounds_used)

        # Ažuriraj povijest za pravilo o 2 tipa suhih guma
        if weather in (2, 3):
            self.wet_or_damp_seen = True
        if comp in (0, 1, 2):
            self.dry_compounds_used.add(int(comp))

        #reward = 100 - lap_time * 0.5 + 0.3 * tyre_life
        # promjena reward funkcije tj. njena normalizacija
        reward = (100 - lap_time * 0.5 + 0.3 * tyre_life) / 10.0

        # PIT logika
        # if pit == 1:
        #     reward += 4.0 if tyre_life < 10 or weather > 1 else -2.0
        # if pit == 0 and tyre_life < 5:
        #     reward -= 4.0

        # PIT logika (smanjeno)
        if pit == 1:
            reward += 1.5 if (tyre_life < 10 or weather > 1) else -1.0
        if pit == 0 and tyre_life < 5:
            reward -= 3.0

        # STIL vožnje
        if style == 0 and front_gap < 1.5:
            reward += 2.0
        if style == 2 and tyre_life > 20:
            reward -= 1.5

        # === Safety Car / VSC / crveni / žuti (TrackStatus) ===
        # FastF1 kodovi: 1=čisto, 2=žuti, 4=SC, 5=crveni, 6=VSC, 7=kraj VSC-a
        caution = track_status in (2, 4, 5, 6, 7)
        if caution and style == 2:
            reward -= 2.0  # pretjecanje je zabranjeno pod caution uvjetima
        if track_status in (4, 6) and pit == 1:
            reward += 1.5  # "besplatan" pit prozor pod SC/VSC - realan poticaj
        if track_status == 5 and pit == 1:
            reward += 1.0  # crveni: promjena guma bez normalnog gubitka vremena

        # === Evaluacija compounda ===
        # NAPOMENA: prijašnja verzija je davala fiksnu nagradu +1.5 za SVAKI
        # odabir SOFT-a kad je suho (weather==1), i kaznu -1.5 za bilo koji
        # drugi compound. Dok je Weather bio nasumičan (~60% suho), taj efekt
        # se razrijedio pa nije bio presudan. Otkad Weather dolazi iz stvarnih
        # FastF1 podataka, SVE tri utrke u datasetu su potpuno suhe (weather==1
        # doslovno uvijek) - pa je ta stara linija de facto postala
        # "uvijek nagradi SOFT, uvijek kazni MEDIUM/HARD", što je pogrešno:
        # stvarna F1 strategija zahtijeva IZMJENU SOFT/MEDIUM/HARD kroz utrku,
        # ne stalno isti compound. Zato ovdje kažnjavamo samo OČITO pogrešne
        # izbore (INTER/WET na suhoj stazi), a odabir SOFT vs MEDIUM vs HARD
        # prepuštamo signalu podudaranja sa stvarnom gumom vozača (true_comp,
        # niže) i R10-stil poticaju za raznolikost odmah ispod.
        if weather == 1 and comp in (3, 4):
            reward -= 2.5  # INTER/WET na suhom nema smisla
        elif weather == 2 and comp != 3:
            reward -= 1.5
        elif weather == 3 and comp != 4:
            reward -= 2.5

        # Dodatna kazna za pogrešan compound kod kiše
        if weather == 3 and comp != 4:
            reward -= 2.5  # dodatna penalizacija

        # Gusti (svaki-korak) poticaj za stvarnu raznolikost suhih guma tijekom
        # suhe utrke - zamjena za staru "Promjena compounda po lošem vremenu"
        # koja je ovisila o weather>1 i sad je praktički mrtav kod za ove 3
        # utrke. Ako agent uvede NOVI (dosad nekorišten) suhi compound dok su
        # gume dosta istrošene, to je vjerojatno prava strateška promjena -
        # nagradi je, umjesto da čekamo samo terminalnu kaznu na kraju epizode.
        if (
            not self.wet_or_damp_seen
            and comp in (0, 1, 2)
            and comp not in self._dry_compounds_before_this_lap
            and tyre_life < 15
        ):
            reward += 2.5

        # Promjena compounda po lošem vremenu
        # NAPOMENA (popravak): self.previous_actions[-1] je TRENUTNA akcija
        # (upravo dodana par redaka iznad) - usporedba s njom samom sobom je
        # uvijek False, pa je pozitivna grana ispod bila mrtav kod. Ispravno
        # je usporediti s [-2] (akcija iz PRETHODNOG kruga).
        if len(self.previous_actions) > 1 and weather > 1:
            if comp != self.previous_actions[-2][1]:
                reward += 1.0
            else:
                reward -= 0.5

        # NAPOMENA (popravak): ovdje je ranije stajalo "reward += np.sqrt(self.index)"
        # ("Stabilizacija rasta kroz epizodu") - bezuvjetan bonus koji raste sa
        # svakim krugom NEOVISNO o akciji. Pred kraj epizode (krug ~650) taj
        # bonus iznosi +25/korak - višestruko više od bilo koje nagrade za
        # ispravnu odluku (npr. +4 za pogođen compound). Analiza raspisa
        # reward komponenti pokazala je da ovaj JEDAN član nosi ~8000-13000
        # od ukupnog reward-a kroz epizodu, dok SVE odluke zajedno (pit,
        # compound, stil) nose tek nekoliko stotina. To je agentu davalo
        # snažan poticaj da jednostavno "preživi što duže" (izbjegne rani
        # DNF) umjesto da uči suptilnu, stanje-ovisnu strategiju - najviše je
        # pogodilo baš stil vožnje, jedinu akciju koja izravno utječe na
        # vjerojatnost DNF-a. Uklonjeno.

        # Kazna ako nikad nije pitted / ako nisu korištena 2 tipa suhih guma -
        # provjerava se na kraju SVAKE utrke unutar epizode (race_end_for), ne
        # samo na kraju cijelog spojenog niza od 12 utrka (ranija greška: ta
        # bi kazna praktički stigla ispitati samo posljednju od 12 utrka).
        if self.index == self.race_end_for[self.index]:
            if 1 not in [a[0] for a in self.previous_actions]:
                reward -= 7.0

            # Kazna ako je TA utrka bila suha, a agent je do kraja
            # koristio samo JEDAN tip suhih guma (FIA: obavezna 2 tipa
            # suhih guma po suhoj utrci) - isto pravilo koje safety_layer.py
            # kasnije prisilno korigira; ovdje ga agent uči izbjegavati sam.
            if not self.wet_or_damp_seen and len(self.dry_compounds_used) < 2:
                reward -= 8.0

        # Random penal za riskantno ponašanje (ostaje) - ako se dogodi, agent je
        # "otpao" (DNF) iz TE utrke, ali epizoda se nastavlja na SLJEDEĆOJ utrci
        # umjesto da se cijela epizoda prekine (ranija verzija je prekidala SVE
        # preostale utrke - otkad je weather>1 puno češći zbog augmentacije,
        # to je prečesto sprječavalo agenta da uopće stigne do kasnijih utrka
        # u istoj epizodi).
        if style == 2 and weather > 1 and self._rng.random() < 0.1:
            reward -= 10.0
            race_end = self.race_end_for[self.index]
            if race_end >= len(self.filtered_obs_array) - 1:
                # ovo je bila zadnja utrka u epizodi - epizoda stvarno završava
                return scale_observation(self.filtered_obs_array[self.index]), reward, True, False, {}
            # "DNF" u ovoj utrci - skoči na početak SLJEDEĆE utrke, epizoda traje dalje
            self.index = race_end + 1
            self.wet_or_damp_seen = False
            self.dry_compounds_used = set()
            self.previous_actions = []
            self.lap_in_race = 0
            return scale_observation(self.filtered_obs_array[self.index]), reward, False, False, {}

        # Penal za monotonost strategije (koristi lap_in_race - brojač PO UTRCI,
        # ne globalni self.index, da se ne uspoređuju krugovi iz dvije različite
        # utrke odmah nakon granice)
        if self.lap_in_race > 3:
            recent = self.previous_actions[-3:]
            if len(set([x[1] for x in recent])) == 1:
                reward -= 3.0
            if len(set([x[2] for x in recent])) == 1:
                reward -= 1.5
            if len(set([x[0] for x in recent])) == 1:
                reward -= 1.0

        if self.lap_in_race > 5:
            recent_styles = [x[2] for x in self.previous_actions[-5:]]
            if len(set(recent_styles)) == 1:
                reward -= 3.0

        if self.lap_in_race > 10:
            last_10 = self.previous_actions[-10:]
            unique_strategies = len(set(tuple(x) for x in last_10))
            if unique_strategies >= 5:
                reward += 3.0

        # Dodatne penalizacije/bonusi po vremenu
        if weather == 2:
            if comp != 3:
                reward -= 1.5
            if style == 2:
                reward -= 1.5

        # (Napomena: ranije je ovdje postojala linija koja je davala +1.0 za
        # kombinaciju suho+SOFT+agresivno bez obzira na kontekst - uklonjena je
        # jer je, u kombinaciji s gore uklonjenom "uvijek SOFT" nagradom, gurala
        # politiku prema stalnom SOFT+agresivno ponašanju čim je Weather postao
        # realan i uvijek suh. Odabir stila sad se oslanja na R6/R7/R9-stil
        # kazne iznad i na podudaranje sa stvarnim stilom vozača ispod.)

        # Pojačana važnost preciznosti (ravnoteža između comp i style)
        # VAŽNO: self.true_actions je indeksiran prema IZVORNOM (neizmiješanom)
        # redoslijedu obs_array, a self.index je pozicija u PROMIJEŠANOM
        # filtered_obs_array - zato koristimo self.orig_indices za ispravan
        # lookup (stara verzija je ovdje imala latentnu grešku: koristila je
        # self.index izravno, pa je nakon miješanja redaka uspoređivala
        # akciju s ground-truth akcijom SASVIM DRUGOG kruga).
        true_idx = self.orig_indices[self.index]
        if self.true_actions is not None and true_idx < len(self.true_actions):
            true_pit, true_comp, true_style = self.true_actions[true_idx]
            # NAPOMENA (popravak v1): stvarna stopa pit stopova je samo ~2.5% -
            # simetrična nagrada ±2.0 činila je "uvijek predvidi 0" statistički
            # najisplativijom strategijom, pa je RAW politika ostala zaglavljena
            # na pit=0. Riješeno asimetričnom težinom.
            # NAPOMENA (popravak v2): prvotna asimetrija (+6/-4 vs +1/-1, omjer
            # penala 4:1) bila je PREJAKA - agent je počeo "bacati široku mrežu"
            # i pitati 14-29% vremena (SAFE izlaz) naspram stvarnih 2.5%. Blaže
            # omjer (+4/-3 vs +1/-1.5, omjer penala 2:1) i dalje jasno favorizira
            # pogađanje rijetkog "treba pitati" trenutka, ali uz veću kaznu za
            # lažni pozitivan da se ne isplati pitati "za svaki slučaj".
            if true_pit == 1:
                reward += 4.0 if pit == 1 else -3.0
            else:
                reward += 1.0 if pit == 0 else -1.5
            # Ako je krug sintetički prebojen u kišu/vlagu, stvarni vozač
            # nikad nije vozio pod tim uvjetima - njegova stvarna (suha) guma
            # tu više nije ispravan cilj, pa tu komponentu isključujemo i
            # prepuštamo je pravilima (R1/R2 gore) umjesto proturječnom
            # dvostrukom signalu.
            if not is_synth_weather_lap:
                reward += 4.0 if comp == true_comp else -1.5 # pojacano sa 3.o na 4.o i -1.0 na -1.5
            if not (is_synth_weather_lap or is_synth_caution_lap):
                reward += 4.0 if style == true_style else -1.5 # isto pojacano kao prethodno

        # Blagi dnevni penal
        reward -= 0.5

        self.lap_in_race += 1

        # Kraj epizode
        done = self.index >= len(self.filtered_obs_array) - 1
        if not done:
            self.index += 1
            obs = self.filtered_obs_array[self.index]
        else:
            obs = self.filtered_obs_array[-1]

        return scale_observation(obs), reward, done, False, {}
